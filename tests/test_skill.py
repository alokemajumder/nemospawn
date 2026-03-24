"""Tests for reusable agent skill."""

from unittest.mock import patch
from pathlib import Path

from nemospawn.core.skill import (
    install_skill, uninstall_skill, is_installed, SKILL_CONTENT,
)


def test_install_skill_claude(tmp_path):
    skill_dir = tmp_path / ".claude" / "skills" / "nemospawn"
    with patch("nemospawn.core.skill.SKILL_DIR_CLAUDE", skill_dir):
        installed = install_skill(["claude"])
        assert len(installed) == 1
        assert skill_dir.exists()
        assert (skill_dir / "skill.md").exists()

        content = (skill_dir / "skill.md").read_text()
        assert "NemoSpawn Agent Skill" in content
        assert "nemospawn task" in content
        assert "nemospawn inbox" in content
        assert "nemospawn plan" in content
        assert "nemospawn lifecycle" in content


def test_install_skill_all(tmp_path):
    claude_dir = tmp_path / ".claude" / "skills" / "nemospawn"
    codex_dir = tmp_path / ".codex" / "skills" / "nemospawn"
    with patch("nemospawn.core.skill.SKILL_DIR_CLAUDE", claude_dir), \
         patch("nemospawn.core.skill.SKILL_DIR_CODEX", codex_dir):
        installed = install_skill(["claude", "codex"])
        assert len(installed) == 2
        assert claude_dir.exists()
        assert codex_dir.exists()


def test_uninstall_skill(tmp_path):
    skill_dir = tmp_path / ".claude" / "skills" / "nemospawn"
    with patch("nemospawn.core.skill.SKILL_DIR_CLAUDE", skill_dir):
        install_skill(["claude"])
        assert (skill_dir / "skill.md").exists()

        removed = uninstall_skill(["claude"])
        assert len(removed) == 1
        assert not (skill_dir / "skill.md").exists()


def test_is_installed(tmp_path):
    skill_dir = tmp_path / ".claude" / "skills" / "nemospawn"
    with patch("nemospawn.core.skill.SKILL_DIR_CLAUDE", skill_dir):
        assert is_installed("claude") is False
        install_skill(["claude"])
        assert is_installed("claude") is True


def test_uninstall_not_installed(tmp_path):
    skill_dir = tmp_path / ".claude" / "skills" / "nemospawn"
    with patch("nemospawn.core.skill.SKILL_DIR_CLAUDE", skill_dir):
        removed = uninstall_skill(["claude"])
        assert removed == []


def test_skill_content_has_key_commands():
    assert "NEMOSPAWN_TEAM" in SKILL_CONTENT
    assert "NEMOSPAWN_AGENT" in SKILL_CONTENT
    assert "nemospawn task list" in SKILL_CONTENT
    assert "nemospawn inbox receive" in SKILL_CONTENT
    assert "nemospawn plan submit" in SKILL_CONTENT
    assert "nemospawn lifecycle idle" in SKILL_CONTENT
    assert "nemospawn artifact register" in SKILL_CONTENT
