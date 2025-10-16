from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def render_template(**context):
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("keywords.html")
    return template.render(**context)


def test_keywords_template_contains_single_project_select(tmp_path):
    html = render_template(projects=[{"id": "proj-1", "name": "Project One"}], selected_project="proj-1")
    assert html.count('id="project"') == 1
    assert 'id="keyword-progress"' in html
    assert 'id="insight-list"' in html
