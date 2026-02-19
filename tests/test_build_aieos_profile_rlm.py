"""Unit tests for build_aieos_profile Scene and create_scene_from_panels."""

from build_aieos_profile_rlm import Scene, create_scene_from_panels


class TestScene:
    """Tests for Scene dataclass with preserved panel structure."""

    def test_scene_id(self):
        """Test scene_id property."""
        scene = Scene(
            issue="pkna-3",
            page_numbers=[45, 46, 47],
            panels=[{"description": "Panel 1", "dialogues": []}],
        )
        assert scene.scene_id == "pkna-3_45"

    def test_to_dict_preserves_panels(self):
        """Test to_dict includes full panel data."""
        panels = [
            {
                "is_new_scene": True,
                "description": "Uno's screen lights up",
                "caption_text": "In the tower...",
                "dialogues": [
                    {"character": "Uno", "line": "Benvenuto!"},
                    {"character": "Paperinik", "line": "Ciao Uno."},
                ],
            },
            {
                "is_new_scene": False,
                "description": "Close-up of Uno",
                "caption_text": None,
                "dialogues": [{"character": "Uno", "line": "Come posso aiutarti?"}],
            },
        ]
        scene = Scene(issue="pkna-0", page_numbers=[28, 29], panels=panels)

        d = scene.to_dict()

        assert d["scene_id"] == "pkna-0_28"
        assert d["issue"] == "pkna-0"
        assert d["page_numbers"] == [28, 29]
        assert d["panels"] == panels
        assert len(d["panels"]) == 2
        # Verify panel structure is preserved
        assert d["panels"][0]["caption_text"] == "In the tower..."
        assert len(d["panels"][0]["dialogues"]) == 2

    def test_get_uno_dialogues(self):
        """Test extracting Uno's dialogue lines."""
        panels = [
            {
                "dialogues": [
                    {"character": "Uno", "line": "Prima battuta."},
                    {"character": "Paperinik", "line": "Risposta."},
                ]
            },
            {
                "dialogues": [
                    {"character": "Uno", "line": "Seconda battuta."},
                ]
            },
            {"dialogues": []},
        ]
        scene = Scene(issue="pkna-1", page_numbers=[10], panels=panels)

        uno_lines = scene.get_uno_dialogues()

        assert uno_lines == ["Prima battuta.", "Seconda battuta."]

    def test_get_uno_dialogues_case_insensitive(self):
        """Test that character matching is case-insensitive."""
        panels = [
            {
                "dialogues": [
                    {"character": "UNO", "line": "Line 1"},
                    {"character": "uno", "line": "Line 2"},
                    {"character": "Uno", "line": "Line 3"},
                ]
            }
        ]
        scene = Scene(issue="pkna-0", page_numbers=[5], panels=panels)

        assert scene.get_uno_dialogues() == ["Line 1", "Line 2", "Line 3"]

    def test_get_other_characters(self):
        """Test extracting other character names."""
        panels = [
            {
                "dialogues": [
                    {"character": "Uno", "line": "Hello"},
                    {"character": "Paperinik", "line": "Hi"},
                    {"character": "Everett", "line": "Greetings"},
                ]
            },
            {
                "dialogues": [
                    {"character": "Paperinik", "line": "Another line"},
                ]
            },
        ]
        scene = Scene(issue="pkna-2", page_numbers=[15], panels=panels)

        others = scene.get_other_characters()

        assert others == {"Paperinik", "Everett"}

    def test_get_other_characters_empty(self):
        """Test get_other_characters with only Uno speaking."""
        panels = [
            {
                "dialogues": [
                    {"character": "Uno", "line": "Solo monologue"},
                ]
            }
        ]
        scene = Scene(issue="pkna-0", page_numbers=[1], panels=panels)

        assert scene.get_other_characters() == set()

    def test_has_uno_true(self):
        """Test has_uno returns True when Uno has dialogue."""
        panels = [{"dialogues": [{"character": "Uno", "line": "Present!"}]}]
        scene = Scene(issue="pkna-0", page_numbers=[1], panels=panels)

        assert scene.has_uno() is True

    def test_has_uno_false(self):
        """Test has_uno returns False when Uno has no dialogue."""
        panels = [{"dialogues": [{"character": "Paperinik", "line": "Where is Uno?"}]}]
        scene = Scene(issue="pkna-0", page_numbers=[1], panels=panels)

        assert scene.has_uno() is False


class TestCreateSceneFromPanels:
    """Tests for create_scene_from_panels function."""

    def test_creates_scene_when_uno_present(self):
        """Test scene is created when Uno appears in dialogues."""
        panels = [
            {
                "is_new_scene": True,
                "description": "Tower interior",
                "dialogues": [
                    {"character": "Uno", "line": "Ciao!"},
                ],
            }
        ]

        scene = create_scene_from_panels("pkna-0", [28], panels)

        assert scene is not None
        assert scene.issue == "pkna-0"
        assert scene.page_numbers == [28]
        assert scene.panels == panels

    def test_returns_none_when_uno_absent(self):
        """Test None is returned when Uno is not in the scene."""
        panels = [
            {
                "dialogues": [
                    {"character": "Paperinik", "line": "Where is Uno?"},
                    {"character": "Everett", "line": "I don't know."},
                ],
            }
        ]

        scene = create_scene_from_panels("pkna-0", [10], panels)

        assert scene is None

    def test_preserves_all_panel_fields(self):
        """Test that all panel fields are preserved in the scene."""
        panels = [
            {
                "is_new_scene": True,
                "description": "Detailed panel description",
                "caption_text": "The next day...",
                "dialogues": [
                    {"character": "Uno", "line": "Good morning."},
                ],
            }
        ]

        scene = create_scene_from_panels("pkna-1", [5, 6], panels)

        assert scene is not None
        assert scene.panels[0]["is_new_scene"] is True
        assert scene.panels[0]["description"] == "Detailed panel description"
        assert scene.panels[0]["caption_text"] == "The next day..."
