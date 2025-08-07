import json
import os
import streamlit as st
import sys
import difflib
from PIL import Image
from typing import List, Dict, Any


def load_evaluation_data(file_path: str) -> List[Dict[str, Any]]:
    """Load evaluation data from JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def get_score_color(score: float) -> str:
    """Get color based on score value."""
    if score >= 0.8:
        return "green"
    elif score >= 0.5:
        return "orange"
    else:
        return "red"


def format_dialogue_item(item: Dict[str, str]) -> str:
    """Format a dialogue item for display."""
    return f"**{item['character']}**: {item['text']}"


def create_dialogue_diff(expected: List[Dict[str, str]], predicted: List[Dict[str, str]]) -> str:
    """Create a visual diff between expected and predicted dialogues, highlighting differences at character level."""
    
    if not expected and not predicted:
        return "✅ **Perfect Match!**"
    
    if not expected:
        return "**All items are extra predictions**"
    
    if not predicted:
        return "**All expected items are missing**"
    
    results = []
    max_len = max(len(expected), len(predicted))
    
    # Compare items side by side
    for i in range(max_len):
        exp_item = expected[i] if i < len(expected) else None
        pred_item = predicted[i] if i < len(predicted) else None
        
        if exp_item is None and pred_item is not None:
            # Extra prediction
            results.append(f"**[{i+1}] Extra**: :blue[**{pred_item['character']}**: {pred_item['text']}]")
        elif pred_item is None and exp_item is not None:
            # Missing expected item
            results.append(f"**[{i+1}] Missing**: :red[**{exp_item['character']}**: {exp_item['text']}]")
        elif exp_item is not None and pred_item is not None:
            # Compare character and text
            char_match = exp_item['character'] == pred_item['character']
            text_match = exp_item['text'].strip() == pred_item['text'].strip()
            
            if char_match and text_match:
                results.append(f"**[{i+1}] Perfect Match**: :green[**{exp_item['character']}**: {exp_item['text']}]")
            elif text_match and not char_match:
                results.append(f"**[{i+1}] Text OK, Wrong Character**:")
                results.append(f"  Expected: :red[**{exp_item['character']}**]: {exp_item['text']}")
                results.append(f"  Got: :blue[**{pred_item['character']}**]: {pred_item['text']}")
            elif char_match and not text_match:
                results.append(f"**[{i+1}] Character OK, Wrong Text**: **{exp_item['character']}**")
                
                # Character-level diff for text
                exp_text = exp_item['text']
                pred_text = pred_item['text']
                
                # Use difflib for character-level comparison
                diff = list(difflib.ndiff([exp_text], [pred_text]))
                
                if len(diff) >= 2:
                    exp_line = diff[0][2:] if diff[0].startswith('- ') else exp_text
                    pred_line = diff[1][2:] if diff[1].startswith('+ ') else pred_text
                    
                    results.append(f"  Expected: :red[{exp_line}]")
                    results.append(f"  Got: :blue[{pred_line}]")
                else:
                    results.append(f"  Expected: :red[{exp_text}]")
                    results.append(f"  Got: :blue[{pred_text}]")
            else:
                # Both character and text are wrong
                results.append(f"**[{i+1}] Complete Mismatch**:")
                results.append(f"  Expected: :red[**{exp_item['character']}**: {exp_item['text']}]")
                results.append(f"  Got: :blue[**{pred_item['character']}**: {pred_item['text']}]")
    
    return "\n\n".join(results) if results else "✅ **Perfect Match!**"


def create_side_by_side_comparison(expected: List[Dict[str, str]], predicted: List[Dict[str, str]]) -> tuple:
    """Create side-by-side comparison of dialogues."""
    expected_text = "\n\n".join(
        [format_dialogue_item(item) for item in expected])
    predicted_text = "\n\n".join(
        [format_dialogue_item(item) for item in predicted])
    return expected_text, predicted_text


def calculate_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for the evaluation data."""
    scores = [item['score'] for item in data]
    return {
        'total': len(data),
        'avg_score': sum(scores) / len(scores) if scores else 0,
        'min_score': min(scores) if scores else 0,
        'max_score': max(scores) if scores else 0,
        'high_scores': len([s for s in scores if s >= 0.8]),
        'medium_scores': len([s for s in scores if 0.5 <= s < 0.8]),
        'low_scores': len([s for s in scores if s < 0.5])
    }


def main():
    if len(sys.argv) < 2:
        st.error("Usage: streamlit run review-extract-eval.py <eval_jsonl_path>")
        st.stop()

    eval_path = sys.argv[1]

    if not os.path.exists(eval_path):
        st.error(f"File not found: {eval_path}")
        st.stop()

    # Load evaluation data
    data = load_evaluation_data(eval_path)

    if not data:
        st.error("No evaluation data found!")
        st.stop()

    # Calculate statistics
    stats = calculate_statistics(data)

    # Streamlit configuration
    st.set_page_config(
        page_title="Evaluation Review Tool",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar for navigation and filtering
    with st.sidebar:
        st.title("📊 Evaluation Review")

        # Statistics
        st.subheader("Statistics")
        st.metric("Total Items", stats['total'])
        st.metric("Average Score", f"{stats['avg_score']:.3f}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Min", f"{stats['min_score']:.3f}")
        with col2:
            st.metric("Max", f"{stats['max_score']:.3f}")

        # Score distribution
        st.subheader("Score Distribution")
        st.success(f"High (≥0.8): {stats['high_scores']}")
        st.warning(f"Medium (0.5-0.8): {stats['medium_scores']}")
        st.error(f"Low (<0.5): {stats['low_scores']}")

        # Filtering
        st.subheader("Filters")
        score_filter = st.selectbox(
            "Filter by Score",
            ["All", "High (≥0.8)", "Medium (0.5-0.8)", "Low (<0.5)"]
        )

        # Apply filter
        filtered_data = data
        if score_filter == "High (≥0.8)":
            filtered_data = [d for d in data if d['score'] >= 0.8]
        elif score_filter == "Medium (0.5-0.8)":
            filtered_data = [d for d in data if 0.5 <= d['score'] < 0.8]
        elif score_filter == "Low (<0.5)":
            filtered_data = [d for d in data if d['score'] < 0.5]

        if not filtered_data:
            st.warning("No items match the current filter.")
            st.stop()

        # Navigation
        st.subheader("Navigation")

        # Initialize navigation index in session state
        if "current_index" not in st.session_state:
            st.session_state.current_index = 0

        # Ensure index is within bounds
        if st.session_state.current_index >= len(filtered_data):
            st.session_state.current_index = 0

        current_index = st.number_input(
            "Item Index",
            min_value=0,
            max_value=len(filtered_data) - 1,
            value=st.session_state.current_index,
            key="nav_index"
        )

        # Update session state when number input changes
        if current_index != st.session_state.current_index:
            st.session_state.current_index = current_index

        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ Previous", disabled=current_index == 0):
                st.session_state.current_index = max(0, current_index - 1)
                st.rerun()

        with col_next:
            if st.button("➡️ Next", disabled=current_index >= len(filtered_data) - 1):
                st.session_state.current_index = min(
                    len(filtered_data) - 1, current_index + 1)
                st.rerun()

    # Main content
    current_item = filtered_data[current_index]

    # Title and score
    st.title("🔍 Evaluation Review Tool")

    # Score display
    score = current_item['score']
    score_color = get_score_color(score)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Current Item",
                  f"{current_index + 1} of {len(filtered_data)}")
    with col2:
        st.markdown(f"**Score**: :{score_color}[{score:.3f}]")
    with col3:
        st.text(f"Image: {current_item['page']}")

    # Main layout
    img_col, diff_col = st.columns([1, 1], gap="large")

    with img_col:
        st.subheader("📖 Comic Page")
        try:
            image = Image.open(current_item['page'])
            st.image(image, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load image: {e}")
            st.text(f"Image path: {current_item['page']}")

    with diff_col:
        st.subheader("📝 Dialogue Comparison")

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Side-by-Side", "Diff View", "Raw JSON"])

        with tab1:
            st.subheader("Expected vs Predicted")
            expected_text, predicted_text = create_side_by_side_comparison(
                current_item['example'],
                current_item['prediction']
            )

            exp_col, pred_col = st.columns(2)
            with exp_col:
                st.markdown("**Expected:**")
                st.markdown(expected_text)
            with pred_col:
                st.markdown("**Predicted:**")
                st.markdown(predicted_text)

        with tab2:
            st.subheader("Differences")
            diff_text = create_dialogue_diff(
                current_item['example'],
                current_item['prediction']
            )
            st.markdown(diff_text)

        with tab3:
            st.subheader("Raw Data")
            with st.expander("Expected", expanded=False):
                st.json(current_item['example'])
            with st.expander("Predicted", expanded=False):
                st.json(current_item['prediction'])
            with st.expander("Full Item", expanded=False):
                st.json(current_item)


if __name__ == "__main__":
    main()
