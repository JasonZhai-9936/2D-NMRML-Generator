"""Entry point for `streamlit run streamlit_app.py` and Streamlit Community Cloud."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from nmr2d.app import main  # noqa: E402

main()
