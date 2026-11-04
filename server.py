"""
Cookidoo MCP Server

Main server file containing MCP tool definitions for interacting with Cookidoo.
"""

import json
import re
import tomllib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from cookidoo_service import CookidooService, is_catalogue_recipe, load_cookidoo_settings
from guided import from_raw
from schemas import CustomRecipe, Step

PROMPTS_DIR = Path(__file__).parent
VERSION = tomllib.loads((PROMPTS_DIR / "pyproject.toml").read_text())["project"]["version"]
READ_ONLY = {"readOnlyHint": True, "openWorldHint": True}
WRITE = {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True}

# Module-level authenticated session, created lazily on first use
_cookidoo_service: CookidooService | None = None


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    try:
        yield
    finally:
        if _cookidoo_service is not None:
            await _cookidoo_service.close()


mcp = FastMCP(
    "cookidoo-mcp-server",
    version=VERSION,
    instructions=(
        "Tools for the Thermomix Cookidoo platform: search recipes, read them, create, "
        "copy, edit or delete the user's own custom recipes, plan recipes in the "
        '"My Week" meal planner, manage the shopping list and recipe collections. '
        "Always show a recipe to the user and get explicit approval before creating or "
        "updating it, and before deleting anything or clearing the shopping list."
    ),
    lifespan=lifespan,
)


def _get_service() -> CookidooService:
    global _cookidoo_service
    if _cookidoo_service is None:
        try:
            _cookidoo_service = CookidooService(load_cookidoo_settings())
        except ValueError as e:
            raise ToolError(f"Configuration error: {e}") from e
    return _cookidoo_service


def _steps(value: list[str | Step] | str) -> list[str | Step]:
    """Accept a list of plain or structured steps, or a string with one step per line."""
    if isinstance(value, str):
        return _lines(value, strip_numbering=True)
    steps: list[str | Step] = []
    for step in value:
        steps += [step] if isinstance(step, Step) else _lines([step], strip_numbering=True)
    return steps


def _lines(value: list[str] | str, strip_numbering: bool = False) -> list[str]:
    """Accept a list, or a string with one item per line (commas are kept)."""
    items = value.split("\n") if isinstance(value, str) else value
    if strip_numbering:
        items = [re.sub(r"^\s*(\d+[.)-]|[-•*])\s*", "", item) for item in items]
    return [item.strip() for item in items if item.strip()]


def _split_list(text: str) -> list[str]:
    """Split on newlines if present, otherwise on commas."""
    parts = text.split("\n") if "\n" in text else text.split(",")
    return [part.strip() for part in parts if part.strip()]


def _ids(value: list[str] | str, what: str = "recipe ID") -> list[str]:
    items = value if isinstance(value, list) else _split_list(value)
    items = [i.strip() for i in items if i.strip()]
    if not items:
        raise ToolError(f"Provide at least one {what}.")
    return items


def _parse_recipe_json(recipe_json: str) -> CustomRecipe:
    try:
        return CustomRecipe.model_validate_json(recipe_json)
    except ValidationError as e:
        raise ToolError(f"Invalid recipe data: {e}") from e


def _day(value: str) -> date:
    if not value:
        return date.today()
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise ToolError(f"Invalid date '{value}', expected YYYY-MM-DD.") from e


def _format_day(day) -> str:
    lines = [f"{day.id}:"]
    lines += [
        f"- {r.name}{' (custom)' if r.is_custom else ''} ({_minutes(r.total_time)})\n"
        f"  ID: {r.id} | {r.url}"
        for r in day.recipes
    ]
    return "\n".join(lines) if day.recipes else f"{day.id}: nothing planned"


def _minutes(seconds: int | float | str | None) -> str:
    # The calendar API returns times as strings ("2400.0"); catalogue recipes as numbers.
    try:
        return f"{round(float(seconds) / 60)} min" if seconds else "?"
    except (TypeError, ValueError):
        return "?"


def _strip_markup(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


@mcp.tool(annotations=READ_ONLY)
async def connect_to_cookidoo() -> str:
    """
    Authenticate with Cookidoo using the credentials from the environment / .env file.

    Optional: every other tool connects automatically on first use. Call this to
    check the configuration or to force a fresh login.
    """
    service = _get_service()
    try:
        await service.login()
    except Exception as e:
        raise ToolError(f"Connection failed: {e}") from e
    return f"Connected to {service.localization_url} as {service.settings.email}"


@mcp.tool(annotations=READ_ONLY)
async def search_recipes(
    query: str = "",
    ingredients: str = "",
    max_total_time: int | None = None,
    max_results: int = 10,
) -> str:
    """
    Search the Cookidoo recipe catalogue.

    Use this to find inspiration or recipes based on what is in the fridge, then pass
    the returned IDs to get_recipe_details.

    Args:
        query: Free-text query (e.g. "risotto", "quick soup")
        ingredients: Ingredients that must be used, comma- or newline-separated
        max_total_time: Maximum total time in minutes
        max_results: Maximum number of recipes to return (default: 10)
    """
    if not (query or ingredients):
        raise ToolError("Provide a query and/or a list of ingredients.")
    try:
        result = await _get_service().search_recipes(
            query=query,
            ingredients=_split_list(ingredients),
            max_total_time=max_total_time,
            max_results=max_results,
        )
    except Exception as e:
        raise ToolError(f"Search failed: {e}") from e

    if not result.recipes:
        return "No recipes found. Try fewer or more common ingredients."
    lines = [f"Found {len(result.recipes)} recipes:", ""]
    for i, hit in enumerate(result.recipes, 1):
        lines.append(f"{i}. {hit.name}\n   ID: {hit.id} | {hit.url}")
    return "\n".join(lines)


@mcp.tool(annotations=READ_ONLY)
async def get_recipe_details(recipe_id: str) -> str:
    """
    Get the full details of a Cookidoo catalogue recipe: ingredients, steps, times.

    Args:
        recipe_id: The Cookidoo recipe ID (e.g. "r59322", "r907015")
    """
    try:
        recipe = await _get_service().get_recipe_details(recipe_id)
    except Exception as e:
        raise ToolError(f"Failed to get recipe details: {e}") from e

    lines = [
        f"Name: {recipe.name}",
        f"ID: {recipe.id}",
        f"Servings: {recipe.serving_size}",
        f"Active time: {_minutes(recipe.active_time)}",
        f"Total time: {_minutes(recipe.total_time)}",
        f"Difficulty: {recipe.difficulty}",
    ]
    if recipe.utensils:
        lines.append(f"Utensils: {', '.join(recipe.utensils)}")

    lines += ["", "Ingredients:"]
    for ingredient in recipe.ingredients:
        line = "  - " + " ".join(p for p in (ingredient.description, ingredient.name) if p)
        if ingredient.preparation:
            line += f", {ingredient.preparation}"
        lines.append(line)

    step_number = 0
    for group in recipe.step_groups:
        lines += ["", f"{group.title}:" if group.title else "Steps:"]
        for step in group.recipe_steps:
            step_number += 1
            lines.append(f"{step_number}. {_strip_markup(step.formatted_text)}")

    if recipe.notes:
        lines += ["", "Notes:", *(f"  - {_strip_markup(note)}" for note in recipe.notes)]
    lines += ["", f"URL: {recipe.url}"]
    return "\n".join(lines)


@mcp.tool(annotations=READ_ONLY)
async def list_custom_recipes() -> str:
    """List the custom recipes created on the user's Cookidoo account."""
    try:
        recipes = await _get_service().list_custom_recipes()
    except Exception as e:
        raise ToolError(f"Failed to list custom recipes: {e}") from e

    if not recipes:
        return "No custom recipes on this account yet."
    return "\n".join(
        f"- {r.name} ({r.serving_size} servings, {_minutes(r.total_time)})\n  ID: {r.id} | {r.url}"
        for r in recipes
    )


@mcp.tool(annotations=READ_ONLY)
async def get_custom_recipe(recipe_id: str) -> str:
    """
    Get one of the user's custom recipes as JSON, in the format expected by
    update_custom_recipe. Use it as the starting point before editing. Steps with
    machine settings or ingredient links come back as structured steps.

    Args:
        recipe_id: The custom recipe ID returned by list_custom_recipes
    """
    try:
        r, instructions = await _get_service().get_custom_recipe(recipe_id)
    except Exception as e:
        raise ToolError(f"Failed to get custom recipe: {e}") from e

    recipe = {
        "name": r.name,
        "ingredients": r.ingredients,
        "steps": [s if isinstance(s, str) else s.model_dump() for s in map(from_raw, instructions)],
        "servings": r.serving_size,
        "prep_time": max(1, round(r.active_time / 60)),
        "total_time": max(1, round(r.total_time / 60)),
        "hints": r.hints or None,
    }
    return f"URL: {r.url}\n\n{json.dumps(recipe, indent=2, ensure_ascii=False)}"


@mcp.tool(annotations=READ_ONLY)
async def generate_recipe_structure(
    name: str,
    ingredients: list[str] | str,
    steps: list[str | Step] | str,
    servings: int = 4,
    prep_time: int = 30,
    total_time: int = 60,
    hints: list[str] | str = "",
) -> str:
    """
    Generate and validate a recipe structure ready for upload to Cookidoo.

    Returns a JSON structure to show to the user and then pass to
    upload_custom_recipe or update_custom_recipe.

    Args:
        name: Recipe name (required)
        ingredients: Ingredients with quantities, as a list (or one per line)
        steps: Cooking steps, as a list (or one per line). A step without machine settings
            or ingredient links is plain text. Otherwise it is an object whose text holds one
            {settings} placeholder per entry of settings; the server writes each setting into
            the text and presets it on the Thermomix. A setting's "mode" picks its fields
            (allowed values are in the schema): "manual" (default: time, temperature, speed,
            reverse), "dough", "browning", "steaming", "turbo", "rice_cooker", "warm_up",
            "blend". "ingredients" links words of the text to an ingredient line. Example:
            {"text": "Add the onion and sauté {settings}.", "settings": [{"time": 180,
            "temperature": "120", "speed": "1"}], "ingredients": [{"text": "the onion",
            "ingredient": "1 onion"}]}
        servings: Number of servings (default: 4, range: 1-20)
        prep_time: Active preparation time in minutes (default: 30)
        total_time: Total time in minutes, at least prep_time (default: 60)
        hints: Optional cooking tips, as a list (or one per line)
    """
    try:
        recipe = CustomRecipe(
            name=name,
            ingredients=_lines(ingredients),
            steps=_steps(steps),
            servings=servings,
            prep_time=prep_time,
            total_time=total_time,
            hints=_lines(hints) or None,
        )
    except ValidationError as e:
        raise ToolError(f"Validation failed: {e}") from e
    return recipe.model_dump_json(indent=2)


@mcp.tool(annotations=WRITE)
async def upload_custom_recipe(recipe_json: str) -> str:
    """
    Create a brand new custom recipe on the user's Cookidoo account.

    Only call this after the user explicitly approved the recipe.

    Args:
        recipe_json: The validated recipe JSON from generate_recipe_structure
    """
    recipe = _parse_recipe_json(recipe_json)
    try:
        created = await _get_service().create_custom_recipe(recipe)
    except Exception as e:
        raise ToolError(f"Upload failed: {e}") from e
    return f"Recipe '{created.name}' created.\nRecipe ID: {created.id}\nURL: {created.url}"


@mcp.tool(annotations={**WRITE, "destructiveHint": True, "idempotentHint": True})
async def update_custom_recipe(recipe_id: str, recipe_json: str) -> str:
    """
    Replace the content of an existing custom recipe on Cookidoo.

    Get the current version with get_custom_recipe, edit it, show the result to the
    user and only call this after their explicit approval.

    Args:
        recipe_id: The custom recipe ID returned by list_custom_recipes
        recipe_json: The full, validated recipe JSON (same format as upload_custom_recipe)
    """
    recipe = _parse_recipe_json(recipe_json)
    try:
        updated = await _get_service().update_custom_recipe(recipe_id, recipe)
    except Exception as e:
        raise ToolError(f"Update failed: {e}") from e
    return f"Recipe '{updated.name}' updated.\nRecipe ID: {updated.id}\nURL: {updated.url}"


@mcp.tool(annotations=WRITE)
async def copy_recipe_to_custom(recipe_id: str, servings: int | None = None) -> str:
    """
    Copy a Cookidoo catalogue recipe into the user's custom recipes, as a starting point
    for their own version (then edit it with get_custom_recipe / update_custom_recipe).

    Args:
        recipe_id: The catalogue recipe ID (e.g. "r59322")
        servings: Number of servings of the copy (default: the original's)
    """
    if not is_catalogue_recipe(recipe_id):
        raise ToolError("Only catalogue recipes (IDs like 'r59322') can be copied.")
    service = _get_service()
    try:
        if servings is None:
            details = await service.get_recipe_details(recipe_id)
            servings = details.serving_size
        created = await service.copy_recipe_to_custom(recipe_id, servings)
    except Exception as e:
        raise ToolError(f"Copy failed: {e}") from e
    return (
        f"Recipe '{created.name}' copied to your recipes.\n"
        f"Recipe ID: {created.id}\nURL: {created.url}"
    )


@mcp.tool(annotations={**WRITE, "destructiveHint": True, "idempotentHint": True})
async def delete_custom_recipe(recipe_id: str) -> str:
    """
    Permanently delete one of the user's custom recipes. This cannot be undone.

    Show the user which recipe will be deleted and only call this after their explicit
    approval.

    Args:
        recipe_id: The custom recipe ID returned by list_custom_recipes
    """
    try:
        await _get_service().delete_custom_recipe(recipe_id)
    except Exception as e:
        raise ToolError(f"Delete failed: {e}") from e
    return f"Custom recipe {recipe_id} deleted."


@mcp.tool(annotations=READ_ONLY)
async def get_shopping_list() -> str:
    """
    Show the Cookidoo shopping list: its recipes, the ingredients still to buy or already
    ticked off, and the additional (free-text) items, with the IDs other tools take.
    """
    try:
        sl = await _get_service().get_shopping_list()
    except Exception as e:
        raise ToolError(f"Failed to load the shopping list: {e}") from e
    if not (sl.recipes or sl.ingredients or sl.additional_items):
        return "The shopping list is empty."

    def item(i) -> str:
        detail = f" {i.description}" if getattr(i, "description", "") else ""
        return f"- [{'x' if i.is_owned else ' '}] {i.name}{detail} (ID: {i.id})"

    sections = []
    if sl.recipes:
        sections.append("Recipes:\n" + "\n".join(f"- {r.name} (ID: {r.id})" for r in sl.recipes))
    if sl.ingredients:
        sections.append("Ingredients:\n" + "\n".join(map(item, sl.ingredients)))
    if sl.additional_items:
        sections.append("Additional items:\n" + "\n".join(map(item, sl.additional_items)))
    return "\n\n".join(sections)


@mcp.tool(annotations=WRITE)
async def add_recipes_to_shopping_list(recipe_ids: list[str] | str) -> str:
    """
    Add the ingredients of recipes to the Cookidoo shopping list. Accepts catalogue and
    custom recipe IDs, mixed freely.

    Args:
        recipe_ids: Recipe IDs, as a list (or comma-separated)
    """
    ids = _ids(recipe_ids)
    try:
        await _get_service().add_recipes_to_shopping_list(ids)
    except Exception as e:
        raise ToolError(f"Failed to add to the shopping list: {e}") from e
    return f"Added the ingredients of {len(ids)} recipe(s) to the shopping list."


@mcp.tool(annotations={**WRITE, "destructiveHint": True, "idempotentHint": True})
async def remove_recipes_from_shopping_list(recipe_ids: list[str] | str) -> str:
    """
    Remove recipes, and their ingredients, from the Cookidoo shopping list.

    Args:
        recipe_ids: Recipe IDs as shown by get_shopping_list, as a list (or comma-separated)
    """
    ids = _ids(recipe_ids)
    try:
        await _get_service().remove_recipes_from_shopping_list(ids)
    except Exception as e:
        raise ToolError(f"Failed to remove from the shopping list: {e}") from e
    return f"Removed {len(ids)} recipe(s) from the shopping list."


@mcp.tool(annotations=WRITE)
async def add_meal_plan_to_shopping_list(day: str = "") -> str:
    """
    Add the ingredients of every recipe planned in a "My Week" calendar week to the
    Cookidoo shopping list.

    Args:
        day: Any day of the week, as YYYY-MM-DD (default: today)
    """
    target = _day(day)
    try:
        names = await _get_service().add_meal_plan_to_shopping_list(target)
    except Exception as e:
        raise ToolError(f"Failed to add the meal plan to the shopping list: {e}") from e
    if not names:
        return "Nothing planned that week; the shopping list is unchanged."
    return "Added to the shopping list:\n" + "\n".join(f"- {n}" for n in names)


@mcp.tool(annotations=WRITE)
async def add_shopping_list_items(items: list[str] | str) -> str:
    """
    Add free-text items (not tied to a recipe, e.g. "kitchen paper") to the shopping list.

    Args:
        items: Item names, as a list (or one per line / comma-separated)
    """
    names = _ids(items, "item")
    try:
        added = await _get_service().add_shopping_list_items(names)
    except Exception as e:
        raise ToolError(f"Failed to add items: {e}") from e
    return "Added:\n" + "\n".join(f"- {i.name} (ID: {i.id})" for i in added)


@mcp.tool(annotations={**WRITE, "destructiveHint": True, "idempotentHint": True})
async def remove_shopping_list_items(item_ids: list[str] | str) -> str:
    """
    Remove additional (free-text) items from the shopping list. Recipe ingredients are
    removed with remove_recipes_from_shopping_list instead.

    Args:
        item_ids: Additional item IDs as shown by get_shopping_list
    """
    ids = _ids(item_ids, "item ID")
    try:
        await _get_service().remove_shopping_list_items(ids)
    except Exception as e:
        raise ToolError(f"Failed to remove items: {e}") from e
    return f"Removed {len(ids)} item(s)."


@mcp.tool(annotations={**WRITE, "idempotentHint": True})
async def mark_shopping_list_items(item_ids: list[str] | str, owned: bool = True) -> str:
    """
    Tick shopping list items off as already owned / bought (or untick them with
    owned=false). Works for both ingredients and additional items.

    Args:
        item_ids: Item IDs as shown by get_shopping_list
        owned: True to tick off, False to put back on the list
    """
    ids = _ids(item_ids, "item ID")
    try:
        missing = await _get_service().mark_shopping_list_items(ids, owned)
    except Exception as e:
        raise ToolError(f"Failed to update items: {e}") from e
    done = len(ids) - len(missing)
    text = f"{'Ticked off' if owned else 'Unticked'} {done} item(s)."
    if missing:
        text += f"\nNot on the shopping list: {', '.join(missing)}"
    return text


@mcp.tool(annotations={**WRITE, "destructiveHint": True, "idempotentHint": True})
async def clear_shopping_list() -> str:
    """
    Empty the whole shopping list: recipes, ingredients and additional items.

    Only call this after the user explicitly asked for it.
    """
    try:
        await _get_service().clear_shopping_list()
    except Exception as e:
        raise ToolError(f"Failed to clear the shopping list: {e}") from e
    return "The shopping list is now empty."


def _format_collection(c) -> str:
    recipes = [r for chapter in c.chapters for r in chapter.recipes]
    lines = [f"{c.name} ({len(recipes)} recipes) | ID: {c.id}"]
    lines += [f"  - {r.name} ({_minutes(r.total_time)}) | ID: {r.id}" for r in recipes]
    return "\n".join(lines)


@mcp.tool(annotations=READ_ONLY)
async def list_collections() -> str:
    """
    List the user's own recipe collections and the Cookidoo collections they follow,
    with their recipes.
    """
    try:
        cols = await _get_service().list_collections()
    except Exception as e:
        raise ToolError(f"Failed to list collections: {e}") from e
    sections = [
        "My collections:\n" + ("\n".join(map(_format_collection, cols.custom)) or "(none)"),
        "Followed Cookidoo collections:\n"
        + ("\n".join(map(_format_collection, cols.managed)) or "(none)"),
    ]
    return "\n\n".join(sections)


@mcp.tool(annotations=WRITE)
async def create_collection(name: str, recipe_ids: list[str] | str = "") -> str:
    """
    Create a new collection of the user's, optionally filled with catalogue recipes.

    Args:
        name: The collection name
        recipe_ids: Optional catalogue recipe IDs to put in it
    """
    ids = _catalogue_ids(recipe_ids) if recipe_ids else []
    service = _get_service()
    try:
        collection = await service.create_collection(name)
        if ids:
            collection = await service.add_recipes_to_collection(collection.id, ids)
    except Exception as e:
        raise ToolError(f"Failed to create the collection: {e}") from e
    return "Collection created.\n" + _format_collection(collection)


@mcp.tool(annotations={**WRITE, "destructiveHint": True, "idempotentHint": True})
async def delete_collection(collection_id: str) -> str:
    """
    Delete one of the user's own collections (the recipes themselves are kept).

    Only call this after the user's explicit approval.

    Args:
        collection_id: The collection ID from list_collections ("My collections")
    """
    try:
        await _get_service().delete_collection(collection_id)
    except Exception as e:
        raise ToolError(f"Failed to delete the collection: {e}") from e
    return f"Collection {collection_id} deleted."


def _catalogue_ids(recipe_ids: list[str] | str) -> list[str]:
    ids = _ids(recipe_ids)
    custom = [i for i in ids if not is_catalogue_recipe(i)]
    if custom:
        raise ToolError(
            "Collections only hold catalogue recipes (IDs like 'r59322'); "
            f"not allowed: {', '.join(custom)}"
        )
    return ids


@mcp.tool(annotations={**WRITE, "idempotentHint": True})
async def add_recipes_to_collection(collection_id: str, recipe_ids: list[str] | str) -> str:
    """
    Add catalogue recipes to one of the user's own collections.

    Args:
        collection_id: The collection ID from list_collections ("My collections")
        recipe_ids: Catalogue recipe IDs, as a list (or comma-separated)
    """
    ids = _catalogue_ids(recipe_ids)
    try:
        collection = await _get_service().add_recipes_to_collection(collection_id, ids)
    except Exception as e:
        raise ToolError(f"Failed to add to the collection: {e}") from e
    return _format_collection(collection)


@mcp.tool(annotations={**WRITE, "destructiveHint": True, "idempotentHint": True})
async def remove_recipe_from_collection(collection_id: str, recipe_id: str) -> str:
    """
    Remove a recipe from one of the user's own collections.

    Args:
        collection_id: The collection ID from list_collections ("My collections")
        recipe_id: The recipe ID to remove
    """
    try:
        collection = await _get_service().remove_recipe_from_collection(collection_id, recipe_id)
    except Exception as e:
        raise ToolError(f"Failed to remove from the collection: {e}") from e
    return _format_collection(collection)


@mcp.tool(annotations={**WRITE, "idempotentHint": True})
async def follow_collection(collection_id: str) -> str:
    """
    Follow (save) a Cookidoo editorial collection. Its ID is the last part of the
    collection's URL on cookidoo, e.g. "col123456".

    Args:
        collection_id: The Cookidoo collection ID
    """
    try:
        collection = await _get_service().follow_collection(collection_id)
    except Exception as e:
        raise ToolError(f"Failed to follow the collection: {e}") from e
    return "Now following:\n" + _format_collection(collection)


@mcp.tool(annotations={**WRITE, "destructiveHint": True, "idempotentHint": True})
async def unfollow_collection(collection_id: str) -> str:
    """
    Stop following a Cookidoo editorial collection.

    Args:
        collection_id: The collection ID from list_collections ("Followed Cookidoo collections")
    """
    try:
        await _get_service().unfollow_collection(collection_id)
    except Exception as e:
        raise ToolError(f"Failed to unfollow the collection: {e}") from e
    return f"No longer following collection {collection_id}."


@mcp.tool(annotations=READ_ONLY)
async def get_meal_plan(day: str = "") -> str:
    """
    Show the recipes planned in the Cookidoo "My Week" planner for the calendar week
    containing a day.

    Args:
        day: Any day of the week to show, as YYYY-MM-DD (default: today)
    """
    target = _day(day)
    try:
        week = await _get_service().get_meal_plan_week(target)
    except Exception as e:
        raise ToolError(f"Failed to load the meal plan: {e}") from e

    planned = [d for d in week if d.recipes]
    if not planned:
        return "Nothing planned this week."
    return "\n\n".join(_format_day(d) for d in planned)


@mcp.tool(annotations=WRITE)
async def add_to_meal_plan(day: str, recipe_ids: list[str] | str) -> str:
    """
    Plan recipes on a day of the Cookidoo "My Week" planner.

    Accepts catalogue recipe IDs (from search_recipes, e.g. "r59322") and custom recipe
    IDs (from list_custom_recipes), mixed freely.

    Args:
        day: The day to plan the recipes on, as YYYY-MM-DD
        recipe_ids: Recipe IDs, as a list (or comma-separated)
    """
    ids = _ids(recipe_ids)
    target = _day(day)
    try:
        planned = await _get_service().add_to_meal_plan(target, ids)
    except Exception as e:
        raise ToolError(f"Failed to add to the meal plan: {e}") from e
    return f"Added {len(ids)} recipe(s).\n\n{_format_day(planned)}"


@mcp.tool(annotations={**WRITE, "destructiveHint": True, "idempotentHint": True})
async def remove_from_meal_plan(day: str, recipe_id: str) -> str:
    """
    Remove a recipe from a day of the Cookidoo "My Week" planner.

    Args:
        day: The day the recipe is planned on, as YYYY-MM-DD
        recipe_id: The catalogue or custom recipe ID, as shown by get_meal_plan
    """
    target = _day(day)
    try:
        planned = await _get_service().remove_from_meal_plan(target, recipe_id.strip())
    except Exception as e:
        raise ToolError(f"Failed to remove from the meal plan: {e}") from e
    return f"Removed {recipe_id}.\n\n{_format_day(planned)}"


@mcp.prompt
def create_thermomix_recipe(dish: str, language: str = "en") -> str:
    """
    Guided workflow to write a Thermomix recipe for a dish and upload it to Cookidoo.

    Args:
        dish: The dish to create, e.g. "Creamy mushroom and parmesan risotto"
        language: "en" or "fr"
    """
    filename = "prompt_FR.md" if language.lower().startswith("fr") else "prompt.md"
    template = (PROMPTS_DIR / filename).read_text(encoding="utf-8")
    return re.sub(r"\[[^\]]*\]", f'"{dish}"', template, count=1)


if __name__ == "__main__":
    mcp.run()
