# Layouts

## `dashboard/app.py`

The Streamlit page shell is created in `main()` with `layout="wide"`. It presents the dashboard title and setup controls, then renders four vertically ordered sections: controller trace, training analysis, advanced neural experiments, and controller performance comparison. Streamlit provides the page chrome; the repository has no separate navigation, sidebar, header, or footer component.
