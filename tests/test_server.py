import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from cookidoo_api import Cookidoo, CookidooLocalizationConfig
from fastmcp import Client
from fastmcp.exceptions import ToolError
from yarl import URL

import server
from cookidoo_service import CookidooService, CookidooSettings, load_cookidoo_settings
from guided import to_instruction
from schemas import CustomRecipe, Step

RECIPE = {
    "name": "Cookies",
    "ingredients": ["200 g flour", "100 g butter"],
    "steps": ["Mix 20 sec/speed 4", "Bake 12 min at 180°C"],
    "servings": 6,
    "prep_time": 15,
    "total_time": 30,
    "hints": None,
}


@pytest.fixture
def service(monkeypatch):
    svc = CookidooService(CookidooSettings(email="a@b.c", password="x"))
    for name in (
        "login",
        "search_recipes",
        "get_recipe_details",
        "list_custom_recipes",
        "get_custom_recipe",
        "create_custom_recipe",
        "update_custom_recipe",
        "get_meal_plan_week",
        "add_to_meal_plan",
        "remove_from_meal_plan",
        "copy_recipe_to_custom",
        "delete_custom_recipe",
        "get_shopping_list",
        "add_recipes_to_shopping_list",
        "remove_recipes_from_shopping_list",
        "add_meal_plan_to_shopping_list",
        "add_shopping_list_items",
        "remove_shopping_list_items",
        "mark_shopping_list_items",
        "clear_shopping_list",
        "list_collections",
        "create_collection",
        "delete_collection",
        "add_recipes_to_collection",
        "remove_recipe_from_collection",
        "follow_collection",
        "unfollow_collection",
    ):
        monkeypatch.setattr(svc, name, AsyncMock())
    monkeypatch.setattr(server, "_cookidoo_service", svc)
    return svc


async def call(tool, **args):
    async with Client(server.mcp) as client:
        result = await client.call_tool(tool, args)
        return result.content[0].text


async def test_all_tools_registered():
    async with Client(server.mcp) as client:
        names = {t.name for t in await client.list_tools()}
        prompts = {p.name for p in await client.list_prompts()}
    assert names == {
        "connect_to_cookidoo",
        "search_recipes",
        "get_recipe_details",
        "list_custom_recipes",
        "get_custom_recipe",
        "generate_recipe_structure",
        "upload_custom_recipe",
        "update_custom_recipe",
        "get_meal_plan",
        "add_to_meal_plan",
        "remove_from_meal_plan",
        "copy_recipe_to_custom",
        "delete_custom_recipe",
        "get_shopping_list",
        "add_recipes_to_shopping_list",
        "remove_recipes_from_shopping_list",
        "add_meal_plan_to_shopping_list",
        "add_shopping_list_items",
        "remove_shopping_list_items",
        "mark_shopping_list_items",
        "clear_shopping_list",
        "list_collections",
        "create_collection",
        "delete_collection",
        "add_recipes_to_collection",
        "remove_recipe_from_collection",
        "follow_collection",
        "unfollow_collection",
    }
    assert prompts == {"create_thermomix_recipe"}


async def test_server_reports_project_version():
    async with Client(server.mcp) as client:
        assert client.server_info.version == server.VERSION == "1.1.0"


async def test_generate_recipe_structure_round_trips_to_upload(service):
    service.create_custom_recipe.return_value = SimpleNamespace(
        name="Cookies", id="01ABC", url="https://cookidoo.fr/created-recipes/fr-FR/01ABC"
    )
    generated = await call(
        "generate_recipe_structure",
        name="Cookies",
        ingredients="200 g flour\n100 g butter",
        steps="1. Mix 20 sec/speed 4\n2) Bake 12 min at 180°C",
        servings=6,
        prep_time=15,
        total_time=30,
    )
    assert json.loads(generated) == RECIPE

    text = await call("upload_custom_recipe", recipe_json=generated)
    assert "01ABC" in text
    service.create_custom_recipe.assert_awaited_once_with(CustomRecipe(**RECIPE))


async def test_generate_keeps_commas_and_accepts_lists(service):
    generated = await call(
        "generate_recipe_structure",
        name="Quiche",
        ingredients=["300 g potatoes, sliced", "2 eggs"],
        steps="1. Slice 5 sec/speed 5\n- Bake 30 min, 180°C",
        hints="Serve warm, with salad",
    )
    recipe = json.loads(generated)
    assert recipe["ingredients"] == ["300 g potatoes, sliced", "2 eggs"]
    assert recipe["steps"] == ["Slice 5 sec/speed 5", "Bake 30 min, 180°C"]
    assert recipe["hints"] == ["Serve warm, with salad"]


async def test_prep_time_cannot_exceed_total_time(service):
    with pytest.raises(ToolError, match="prep_time"):
        await call(
            "generate_recipe_structure",
            name="x",
            ingredients="a",
            steps="b",
            prep_time=40,
            total_time=30,
        )


async def test_upload_rejects_invalid_json(service):
    with pytest.raises(ToolError, match="Invalid recipe data"):
        await call("upload_custom_recipe", recipe_json="{not json")
    service.create_custom_recipe.assert_not_awaited()


async def test_search_splits_ingredients(service):
    service.search_recipes.return_value = SimpleNamespace(
        total=1, recipes=[SimpleNamespace(name="Gratin", id="r1", url="https://x/r1")]
    )
    text = await call("search_recipes", ingredients="chicken, zucchini")
    assert "Gratin" in text and "r1" in text
    assert service.search_recipes.await_args.kwargs["ingredients"] == ["chicken", "zucchini"]


async def test_search_requires_criteria(service):
    with pytest.raises(ToolError):
        await call("search_recipes")


async def test_get_recipe_details_formats_steps_and_times(service):
    service.get_recipe_details.return_value = SimpleNamespace(
        id="r1",
        name="Soup",
        serving_size=4,
        active_time=600,
        total_time=1800,
        difficulty="easy",
        utensils=[],
        ingredients=[SimpleNamespace(name="carrots", description="300 g", preparation="sliced")],
        step_groups=[
            SimpleNamespace(
                title="", recipe_steps=[SimpleNamespace(formatted_text="Cook <b>20 min</b>")]
            )
        ],
        notes=[],
        url="https://x/r1",
    )
    text = await call("get_recipe_details", recipe_id="r1")
    assert "Total time: 30 min" in text
    assert "  - 300 g carrots, sliced" in text
    assert "1. Cook 20 min" in text


def custom_recipe(steps, ingredients):
    """What CookidooService.get_custom_recipe returns: the recipe and its raw steps."""
    recipe = SimpleNamespace(
        name="Soupe",
        ingredients=ingredients,
        serving_size=6,
        active_time=900,
        total_time=1800,
        hints=[],
        url="https://x",
    )
    instructions = [to_instruction(step, "fr-FR") for step in steps]
    return recipe, Cookidoo._process_recipe_steps(instructions, ingredients)


async def test_get_custom_recipe_returns_editable_json(service):
    recipe, raw = custom_recipe(RECIPE["steps"], RECIPE["ingredients"])
    recipe.name = "Cookies"
    service.get_custom_recipe.return_value = (recipe, raw)
    text = await call("get_custom_recipe", recipe_id="01ABC")
    assert json.loads(text.split("\n\n", 1)[1]) == RECIPE


SAUTE = {
    "text": "Faire revenir l'oignon {settings}.",
    "settings": [
        {"mode": "manual", "time": 180, "temperature": "120", "speed": "1", "reverse": False}
    ],
    "ingredients": [{"text": "l'oignon", "ingredient": "1 oignon"}],
}
KNEAD = {
    "text": "Pétrir {settings}.",
    "settings": [{"mode": "dough", "time": 120}],
    "ingredients": [],
}


async def test_generate_accepts_structured_steps(service):
    generated = await call(
        "generate_recipe_structure",
        name="Soupe",
        ingredients=["1 oignon"],
        steps=[
            "1. Éplucher l'oignon.",
            SAUTE,
            {"text": "Pétrir {settings}.", "settings": [{"mode": "dough", "time": 120}]},
        ],
    )
    assert json.loads(generated)["steps"] == ["Éplucher l'oignon.", SAUTE, KNEAD]


@pytest.mark.parametrize(
    "steps",
    [
        [{"text": "Mixer {settings}.", "settings": [{"time": 10, "speed": "12"}]}],
        [{"text": "Mixer {settings}.", "settings": [{"mode": "sous_vide", "time": 10}]}],
        [{"text": "Ajouter le sel.", "ingredients": [{"text": "sel", "ingredient": "sel"}]}],
    ],
)
async def test_generate_rejects_invalid_steps(service, steps):
    with pytest.raises(ToolError):
        await call("generate_recipe_structure", name="Soupe", ingredients=["1 oignon"], steps=steps)


async def test_get_custom_recipe_round_trips_structured_steps(service):
    steps = ["Éplucher.", Step(**SAUTE), Step(**KNEAD)]
    service.get_custom_recipe.return_value = custom_recipe(steps, ["1 oignon"])
    read = (await call("get_custom_recipe", recipe_id="01ABC")).split("\n\n", 1)[1]
    generated = await call(
        "generate_recipe_structure", name="Soupe", ingredients=["1 oignon"], steps=steps
    )
    assert (
        json.loads(read)["steps"] == json.loads(generated)["steps"] == ["Éplucher.", SAUTE, KNEAD]
    )


async def test_service_reads_raw_steps(monkeypatch):
    raw = {
        "recipeId": "01ABC",
        "recipeContent": {
            "name": "Soupe",
            "ingredients": [{"type": "INGREDIENT", "text": "1 oignon"}],
            "instructions": [{"type": "STEP", "text": "Turbo/0.5 sec/3 fois"}],
            "yield": {"value": 4, "unitText": "portion"},
            "prepTime": 600,
            "totalTime": 1800,
        },
    }
    request = AsyncMock(return_value=raw)
    api = SimpleNamespace(
        _ensure_endpoints=AsyncMock(),
        api_endpoint=URL("https://cookidoo.fr"),
        _path=lambda name: "created-recipes/{language}/{id}",
        localization=CookidooLocalizationConfig(country_code="fr", language="fr-FR"),
        _request_json=request,
    )
    svc = CookidooService(CookidooSettings(email="a@b.c", password="x"))
    monkeypatch.setattr(svc, "client", AsyncMock(return_value=api))
    recipe, steps = await svc.get_custom_recipe("01ABC")
    assert recipe.name == "Soupe"
    assert steps == raw["recipeContent"]["instructions"]
    assert str(request.await_args.args[1]) == "https://cookidoo.fr/created-recipes/fr-FR/01ABC"
    assert "customer-recipe.full" in request.await_args.kwargs["headers"]["ACCEPT"]


async def test_service_sends_settings_in_account_language(monkeypatch):
    api = SimpleNamespace(create_custom_recipe=AsyncMock(), update_custom_recipe=AsyncMock())
    svc = CookidooService(CookidooSettings(email="a@b.c", password="x", language="fr-FR"))
    monkeypatch.setattr(svc, "client", AsyncMock(return_value=api))
    recipe = CustomRecipe(name="Soupe", ingredients=["1 oignon"], steps=["Éplucher.", SAUTE])
    await svc.create_custom_recipe(recipe)
    await svc.update_custom_recipe("01ABC", recipe)
    expected = ["Éplucher.", to_instruction(Step(**SAUTE), "fr-FR")]
    assert expected[1].text == "Faire revenir l'oignon 3 min/120°C/vitesse 1."
    assert api.create_custom_recipe.await_args.args[0].instructions == expected
    assert api.update_custom_recipe.await_args.args[1].instructions == expected


async def test_prompt_inserts_dish():
    async with Client(server.mcp) as client:
        result = await client.get_prompt(
            "create_thermomix_recipe", {"dish": "Risotto", "language": "fr"}
        )
    text = result.messages[0].content.text
    assert '"Risotto"' in text and "[" not in text.splitlines()[2]


def test_settings_from_env(monkeypatch):
    monkeypatch.setattr("cookidoo_service.load_dotenv", lambda: None)
    monkeypatch.setenv("COOKIDOO_EMAIL", "a@b.c")
    monkeypatch.setenv("COOKIDOO_PASSWORD", "x")
    monkeypatch.setenv("COOKIDOO_DEVICE", "tm7")
    assert load_cookidoo_settings().device.value == "TM7"

    monkeypatch.setenv("COOKIDOO_DEVICE", "TM99")
    with pytest.raises(ValueError, match="COOKIDOO_DEVICE"):
        load_cookidoo_settings()

    monkeypatch.delenv("COOKIDOO_PASSWORD")
    with pytest.raises(ValueError, match="Missing"):
        load_cookidoo_settings()


def _day(day_id, *recipes):
    return SimpleNamespace(id=day_id, title=day_id, recipes=list(recipes))


def _planned(recipe_id, name, custom=False):
    return SimpleNamespace(
        id=recipe_id, name=name, total_time=1800, is_custom=custom, url=f"https://x/{recipe_id}"
    )


async def test_get_meal_plan_lists_planned_days(service):
    service.get_meal_plan_week.return_value = [
        _day("2026-11-02"),
        _day("2026-11-03", _planned("r1", "Gratin"), _planned("01ABC", "Cookies", custom=True)),
    ]
    text = await call("get_meal_plan", day="2026-11-04")
    service.get_meal_plan_week.assert_awaited_once_with(date(2026, 11, 4))
    assert "2026-11-02" not in text
    assert "2026-11-03:" in text and "Gratin" in text and "Cookies (custom)" in text


async def test_get_meal_plan_accepts_string_times(service):
    # The live calendar API returns total_time as a string such as "2400.0".
    recipe = _planned("r56899", "Leczo")
    recipe.total_time = "2400.0"
    broken = _planned("r1", "Gratin")
    broken.total_time = "n/a"
    service.get_meal_plan_week.return_value = [_day("2026-11-05", recipe, broken)]
    text = await call("get_meal_plan", day="2026-11-05")
    assert "Leczo (40 min)" in text
    assert "Gratin (?)" in text


async def test_get_meal_plan_empty_week(service):
    service.get_meal_plan_week.return_value = [_day("2026-11-02")]
    assert await call("get_meal_plan") == "Nothing planned this week."
    service.get_meal_plan_week.assert_awaited_once_with(date.today())


async def test_get_meal_plan_rejects_bad_date(service):
    with pytest.raises(ToolError, match="YYYY-MM-DD"):
        await call("get_meal_plan", day="next friday")


async def test_add_to_meal_plan_accepts_comma_separated_ids(service):
    service.add_to_meal_plan.return_value = _day("2026-11-05", _planned("r1", "Gratin"))
    text = await call("add_to_meal_plan", day="2026-11-05", recipe_ids="r1, 01ABC")
    service.add_to_meal_plan.assert_awaited_once_with(date(2026, 11, 5), ["r1", "01ABC"])
    assert "Added 2 recipe(s)" in text and "Gratin" in text


async def test_remove_from_meal_plan_reports_empty_day(service):
    service.remove_from_meal_plan.return_value = _day("2026-11-05")
    text = await call("remove_from_meal_plan", day="2026-11-05", recipe_id="r1")
    service.remove_from_meal_plan.assert_awaited_once_with(date(2026, 11, 5), "r1")
    assert "nothing planned" in text


async def test_service_routes_catalogue_and_custom_ids(monkeypatch):
    svc = CookidooService(CookidooSettings(email="a@b.c", password="x"))
    api = AsyncMock()
    api.get_recipes_in_calendar_week.return_value = [_day("2026-11-04"), _day("2026-11-05")]
    monkeypatch.setattr(svc, "client", AsyncMock(return_value=api))
    day = date(2026, 11, 5)

    planned = await svc.add_to_meal_plan(day, ["r59322", "01ABC", "r1"])
    api.add_recipes_to_calendar.assert_awaited_once_with(day, ["r59322", "r1"])
    api.add_custom_recipes_to_calendar.assert_awaited_once_with(day, ["01ABC"])
    assert planned.id == "2026-11-05"

    await svc.remove_from_meal_plan(day, "01ABC")
    api.remove_custom_recipe_from_calendar.assert_awaited_once_with(day, "01ABC")
    api.remove_recipe_from_calendar.assert_not_called()


async def test_copy_recipe_defaults_to_original_servings(service):
    service.get_recipe_details.return_value = SimpleNamespace(serving_size=4)
    service.copy_recipe_to_custom.return_value = SimpleNamespace(
        name="Gratin", id="01NEW", url="https://x/01NEW"
    )
    text = await call("copy_recipe_to_custom", recipe_id="r1")
    service.copy_recipe_to_custom.assert_awaited_once_with("r1", 4)
    assert "01NEW" in text


async def test_copy_recipe_rejects_custom_id(service):
    with pytest.raises(ToolError, match="catalogue"):
        await call("copy_recipe_to_custom", recipe_id="01ABC")
    service.copy_recipe_to_custom.assert_not_called()


async def test_get_shopping_list_formats_sections(service):
    service.get_shopping_list.return_value = SimpleNamespace(
        recipes=[SimpleNamespace(name="Gratin", id="r1")],
        ingredients=[
            SimpleNamespace(id="i1", name="flour", description="200 g", is_owned=True),
            SimpleNamespace(id="i2", name="butter", description="", is_owned=False),
        ],
        additional_items=[SimpleNamespace(id="a1", name="kitchen paper", is_owned=False)],
    )
    text = await call("get_shopping_list")
    assert "- Gratin (ID: r1)" in text
    assert "- [x] flour 200 g (ID: i1)" in text and "- [ ] butter (ID: i2)" in text
    assert "Additional items:\n- [ ] kitchen paper (ID: a1)" in text


async def test_get_shopping_list_empty(service):
    service.get_shopping_list.return_value = SimpleNamespace(
        recipes=[], ingredients=[], additional_items=[]
    )
    assert await call("get_shopping_list") == "The shopping list is empty."


async def test_mark_shopping_list_items_reports_missing(service):
    service.mark_shopping_list_items.return_value = ["zzz"]
    text = await call("mark_shopping_list_items", item_ids="i1, zzz", owned=False)
    service.mark_shopping_list_items.assert_awaited_once_with(["i1", "zzz"], False)
    assert "Unticked 1 item(s)" in text and "zzz" in text


async def test_collections_reject_custom_recipe_ids(service):
    with pytest.raises(ToolError, match="catalogue"):
        await call("add_recipes_to_collection", collection_id="c1", recipe_ids="r1, 01ABC")
    service.add_recipes_to_collection.assert_not_called()


async def test_create_collection_with_recipes(service):
    recipe = SimpleNamespace(id="r1", name="Gratin", total_time=600)
    service.create_collection.return_value = SimpleNamespace(id="c1", name="Xmas", chapters=[])
    service.add_recipes_to_collection.return_value = SimpleNamespace(
        id="c1", name="Xmas", chapters=[SimpleNamespace(recipes=[recipe])]
    )
    text = await call("create_collection", name="Xmas", recipe_ids=["r1"])
    service.add_recipes_to_collection.assert_awaited_once_with("c1", ["r1"])
    assert "Xmas (1 recipes) | ID: c1" in text and "Gratin (10 min)" in text


def _service_with_api(monkeypatch):
    svc = CookidooService(CookidooSettings(email="a@b.c", password="x"))
    api = AsyncMock()
    monkeypatch.setattr(svc, "client", AsyncMock(return_value=api))
    return svc, api


async def test_service_shopping_list_routes_recipe_kinds(monkeypatch):
    svc, api = _service_with_api(monkeypatch)
    await svc.add_recipes_to_shopping_list(["r1", "01ABC"])
    api.add_ingredient_items_for_recipes.assert_awaited_once_with(["r1"])
    api.add_ingredient_items_for_custom_recipes.assert_awaited_once_with(["01ABC"])
    await svc.remove_recipes_from_shopping_list(["01ABC"])
    api.remove_ingredient_items_for_custom_recipes.assert_awaited_once_with(["01ABC"])
    api.remove_ingredient_items_for_recipes.assert_not_called()


async def test_service_mark_items_uses_the_right_endpoint(monkeypatch):
    from cookidoo_api.types import CookidooAdditionalItem, CookidooIngredientItem

    svc, api = _service_with_api(monkeypatch)
    api.get_shopping_list_recipes.return_value = []
    api.get_ingredient_items.return_value = [
        CookidooIngredientItem(id="i1", name="flour", is_owned=False, description="200 g")
    ]
    api.get_additional_items.return_value = [
        CookidooAdditionalItem(id="a1", name="paper", is_owned=False)
    ]
    missing = await svc.mark_shopping_list_items(["i1", "a1", "nope"], True)
    assert missing == ["nope"]
    [ingredients] = api.edit_ingredient_items_ownership.await_args.args
    [additional] = api.edit_additional_items_ownership.await_args.args
    assert [(i.id, i.is_owned) for i in ingredients] == [("i1", True)]
    assert [(i.id, i.is_owned) for i in additional] == [("a1", True)]


async def test_service_meal_plan_to_shopping_list_dedupes(monkeypatch):
    svc, api = _service_with_api(monkeypatch)
    api.get_recipes_in_calendar_week.return_value = [
        _day("2026-11-03", _planned("r1", "Gratin")),
        _day("2026-11-04", _planned("r1", "Gratin"), _planned("01ABC", "Cookies", True)),
    ]
    names = await svc.add_meal_plan_to_shopping_list(date(2026, 11, 4))
    assert names == ["Gratin", "Cookies"]
    api.add_ingredient_items_for_recipes.assert_awaited_once_with(["r1"])
    api.add_ingredient_items_for_custom_recipes.assert_awaited_once_with(["01ABC"])


async def test_service_list_collections_reads_all_pages(monkeypatch):
    svc, api = _service_with_api(monkeypatch)
    api.count_custom_collections.return_value = (3, 2)
    api.get_custom_collections.side_effect = lambda page: [f"c{page}"]
    api.count_managed_collections.return_value = (0, 0)
    cols = await svc.list_collections()
    assert cols.custom == ["c0", "c1"] and cols.managed == []
