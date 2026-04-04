"""Unit tests for extract_emotional.py models and serialization."""

import json
from pathlib import Path

from extract_emotional import (
    SPEECH_ACT_VALUES,
    TONE_VALUES,
    CharacterAppearance,
    DialogueLine,
    ExtractedPage,
    ExtractedPageMeta,
    Extractor,
    IssueSummary,
    Panel,
)


class TestDialogueLine:
    def test_defaults(self):
        dl = DialogueLine(character="Uno", line="Ciao, socio!")
        assert dl.tone == "neutral"
        assert dl.speech_act == "informing"

    def test_all_tones_accepted(self):
        for tone in TONE_VALUES:
            dl = DialogueLine(character="Uno", line="test", tone=tone)
            assert dl.tone == tone

    def test_all_speech_acts_accepted(self):
        for sa in SPEECH_ACT_VALUES:
            dl = DialogueLine(character="Uno", line="test", speech_act=sa)
            assert dl.speech_act == sa

    def test_noncanonical_tone_accepted(self):
        dl = DialogueLine(character="Uno", line="test", tone="excited")
        assert dl.tone == "excited"

    def test_noncanonical_speech_act_accepted(self):
        dl = DialogueLine(character="Uno", line="test", speech_act="singing")
        assert dl.speech_act == "singing"

    def test_full_construction(self):
        dl = DialogueLine(
            character="Paperinik",
            line="Arriviamo!",
            tone="urgent",
            speech_act="commanding",
        )
        assert dl.character == "Paperinik"
        assert dl.line == "Arriviamo!"
        assert dl.tone == "urgent"
        assert dl.speech_act == "commanding"


class TestPanel:
    def test_defaults(self):
        panel = Panel(description="Uno appare sullo schermo.")
        assert panel.is_new_scene is False
        assert panel.caption_text is None
        assert panel.visual_cues == []
        assert panel.dialogues == []

    def test_with_visual_cues(self):
        panel = Panel(
            description="Uno appare sullo schermo.",
            visual_cues=[
                "L'ologramma di Uno trema nervosamente",
                "Lo schermo mostra simboli di allarme",
            ],
        )
        assert len(panel.visual_cues) == 2
        assert "L'ologramma di Uno trema nervosamente" in panel.visual_cues

    def test_with_enriched_dialogues(self):
        panel = Panel(
            description="Uno parla con Paperinik.",
            dialogues=[
                DialogueLine(
                    character="Uno",
                    line="Sai che dispiacere!",
                    tone="sarcastic",
                    speech_act="joking",
                ),
                DialogueLine(
                    character="Paperinik",
                    line="Molto divertente.",
                    tone="playful",
                    speech_act="deflecting",
                ),
            ],
        )
        assert len(panel.dialogues) == 2
        assert panel.dialogues[0].tone == "sarcastic"
        assert panel.dialogues[1].speech_act == "deflecting"

    def test_new_scene_panel(self):
        panel = Panel(
            is_new_scene=True,
            description="Nuova scena nella Ducklair Tower.",
            visual_cues=["La torre è illuminata di notte"],
        )
        assert panel.is_new_scene is True


class TestExtractedPageJsonRoundTrip:
    def _make_page(self) -> ExtractedPage:
        return ExtractedPage(
            summary="Uno avverte Paperinik di un pericolo imminente.",
            panels=[
                Panel(
                    is_new_scene=True,
                    description="Paperinik entra nella Ducklair Tower.",
                    caption_text="Quella notte...",
                    visual_cues=["La torre è avvolta nella nebbia"],
                    dialogues=[
                        DialogueLine(
                            character="Uno",
                            line="Finalmente! Ti stavo aspettando.",
                            tone="affectionate",
                            speech_act="informing",
                        ),
                    ],
                ),
                Panel(
                    description="Uno mostra i dati sullo schermo.",
                    visual_cues=[
                        "Lo schermo mostra una mappa con punti rossi",
                        "L'ologramma di Uno gesticola",
                    ],
                    dialogues=[
                        DialogueLine(
                            character="Uno",
                            line="Abbiamo un problema, socio.",
                            tone="concerned",
                            speech_act="warning",
                        ),
                        DialogueLine(
                            character="Paperinik",
                            line="Che succede?",
                            tone="neutral",
                            speech_act="questioning",
                        ),
                    ],
                ),
            ],
            last_event="Paperinik arriva alla torre",
            characters_introduced=[
                CharacterAppearance(
                    name="Uno",
                    appearance="Ologramma blu di un'intelligenza artificiale, proiettato da uno schermo.",
                ),
                CharacterAppearance(
                    name="Paperinik",
                    appearance="Papero mascherato con costume viola e mantello.",
                ),
            ],
            meta=ExtractedPageMeta(
                model_name="test-model",
                input_page_path="/input/pkna/pkna-0/page_001.jpg",
                lm_usage={"prompt_tokens": 100, "completion_tokens": 200},
            ),
        )

    def test_round_trip(self, tmp_path: Path):
        original = self._make_page()
        json_path = tmp_path / "page.json"
        original.to_json(json_path)
        loaded = ExtractedPage.from_json(json_path)

        assert loaded.summary == original.summary
        assert loaded.last_event == original.last_event
        assert loaded.meta.model_name == original.meta.model_name
        assert loaded.meta.input_page_path == original.meta.input_page_path
        assert loaded.meta.lm_usage == original.meta.lm_usage
        assert loaded.characters_introduced == original.characters_introduced

        assert len(loaded.panels) == len(original.panels)
        for loaded_panel, orig_panel in zip(loaded.panels, original.panels):
            assert loaded_panel.is_new_scene == orig_panel.is_new_scene
            assert loaded_panel.description == orig_panel.description
            assert loaded_panel.caption_text == orig_panel.caption_text
            assert loaded_panel.visual_cues == orig_panel.visual_cues
            assert len(loaded_panel.dialogues) == len(orig_panel.dialogues)
            for loaded_dl, orig_dl in zip(loaded_panel.dialogues, orig_panel.dialogues):
                assert loaded_dl.character == orig_dl.character
                assert loaded_dl.line == orig_dl.line
                assert loaded_dl.tone == orig_dl.tone
                assert loaded_dl.speech_act == orig_dl.speech_act

    def test_json_contains_new_fields(self, tmp_path: Path):
        page = self._make_page()
        json_path = tmp_path / "page.json"
        page.to_json(json_path)

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        panel0 = data["panels"][0]
        assert "visual_cues" in panel0
        assert panel0["visual_cues"] == ["La torre è avvolta nella nebbia"]

        dl0 = panel0["dialogues"][0]
        assert dl0["tone"] == "affectionate"
        assert dl0["speech_act"] == "informing"

    def test_empty_visual_cues_round_trip(self, tmp_path: Path):
        page = ExtractedPage(
            summary="Pagina semplice.",
            panels=[
                Panel(
                    description="Una vignetta vuota.",
                    dialogues=[
                        DialogueLine(character="Uno", line="Niente da segnalare."),
                    ],
                ),
            ],
            last_event=None,
            characters_introduced=[],
            meta=ExtractedPageMeta(
                model_name="test-model",
                input_page_path="/input/page.jpg",
            ),
        )
        json_path = tmp_path / "page.json"
        page.to_json(json_path)
        loaded = ExtractedPage.from_json(json_path)

        assert loaded.panels[0].visual_cues == []
        assert loaded.panels[0].dialogues[0].tone == "neutral"
        assert loaded.panels[0].dialogues[0].speech_act == "informing"


class TestExtractorStateUpdate:
    def _make_summary(self) -> IssueSummary:
        return IssueSummary(
            summary="Test summary",
            key_events=["Event 1", "Event 2"],
            main_characters=["Uno", "Paperinik"],
        )

    def test_update_tracks_characters(self):
        extractor = Extractor(summary=self._make_summary())
        assert extractor.prev_characters == {}

        page = ExtractedPage(
            summary="Page summary",
            panels=[
                Panel(
                    description="Test panel",
                    dialogues=[
                        DialogueLine(
                            character="Uno",
                            line="Ciao!",
                            tone="playful",
                            speech_act="informing",
                        ),
                        DialogueLine(
                            character="Paperinik",
                            line="Ciao Uno!",
                            tone="affectionate",
                            speech_act="informing",
                        ),
                    ],
                ),
            ],
            last_event="Event 1",
            characters_introduced=[
                CharacterAppearance(name="Uno", appearance="Ologramma blu."),
                CharacterAppearance(name="Paperinik", appearance="Papero mascherato."),
            ],
            meta=ExtractedPageMeta(model_name="test", input_page_path="test.jpg"),
        )
        extractor.update_from_extracted(page)

        assert extractor.prev_characters == {
            "Uno": "Ologramma blu.",
            "Paperinik": "Papero mascherato.",
        }
        assert extractor.prev_page_summary == "Page summary"
        assert extractor.prev_event == "Event 1"
        assert extractor.prev_page_panels is not None
        assert len(extractor.prev_page_panels) == 1

    def test_update_accumulates_characters(self):
        extractor = Extractor(summary=self._make_summary())

        page1 = ExtractedPage(
            summary="Page 1",
            panels=[
                Panel(
                    description="Panel 1",
                    dialogues=[
                        DialogueLine(character="Uno", line="Buongiorno."),
                    ],
                ),
            ],
            last_event=None,
            characters_introduced=[
                CharacterAppearance(name="Uno", appearance="Ologramma blu."),
            ],
            meta=ExtractedPageMeta(model_name="test", input_page_path="p1.jpg"),
        )
        extractor.update_from_extracted(page1)
        assert extractor.prev_characters == {"Uno": "Ologramma blu."}

        page2 = ExtractedPage(
            summary="Page 2",
            panels=[
                Panel(
                    description="Panel 2",
                    dialogues=[
                        DialogueLine(character="Due", line="Eccomi."),
                    ],
                ),
            ],
            last_event="Event 2",
            characters_introduced=[
                CharacterAppearance(
                    name="Due", appearance="Copia di Uno, ologramma rosso."
                ),
            ],
            meta=ExtractedPageMeta(model_name="test", input_page_path="p2.jpg"),
        )
        extractor.update_from_extracted(page2)
        assert extractor.prev_characters == {
            "Uno": "Ologramma blu.",
            "Due": "Copia di Uno, ologramma rosso.",
        }
        assert extractor.prev_page_summary == "Page 2"
        assert extractor.prev_event == "Event 2"

    def test_update_preserves_enriched_panels(self):
        extractor = Extractor(summary=self._make_summary())

        page = ExtractedPage(
            summary="Enriched page",
            panels=[
                Panel(
                    description="Panel with cues",
                    visual_cues=["L'ologramma di Uno è rosso"],
                    dialogues=[
                        DialogueLine(
                            character="Uno",
                            line="Allarme!",
                            tone="urgent",
                            speech_act="warning",
                        ),
                    ],
                ),
            ],
            last_event=None,
            characters_introduced=[
                CharacterAppearance(
                    name="Uno", appearance="Ologramma rosso lampeggiante."
                ),
            ],
            meta=ExtractedPageMeta(model_name="test", input_page_path="p.jpg"),
        )
        extractor.update_from_extracted(page)

        assert extractor.prev_page_panels is not None
        panel = extractor.prev_page_panels[0]
        assert panel.visual_cues == ["L'ologramma di Uno è rosso"]
        assert panel.dialogues[0].tone == "urgent"
        assert panel.dialogues[0].speech_act == "warning"


class TestIssueSummaryJsonRoundTrip:
    def test_round_trip(self, tmp_path: Path):
        original = IssueSummary(
            summary="Gli evroniani attaccano.",
            key_events=["Arrivo degli evroniani", "Battaglia finale"],
            main_characters=["Uno", "Paperinik", "Xadhoom"],
        )
        json_path = tmp_path / "summary.json"
        original.to_json(json_path)
        loaded = IssueSummary.from_json(json_path)

        assert loaded.summary == original.summary
        assert loaded.key_events == original.key_events
        assert loaded.main_characters == original.main_characters
