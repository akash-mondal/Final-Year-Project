# NYC UHI Analysis Tool 

Analyze Urban Heat Island (UHI) effects across NYC using a combination of local sensor data, weather station information, and Sentinel-2 satellite imagery. This Streamlit application provides interactive visualizations and AI-powered analysis to understand local heat conditions and potential mitigation strategies.

**Features:**

*   Loads user-provided UHI data  and local weather station data.
*   Fetches Sentinel-2 satellite data for a chosen date via Microsoft Planetary Computer.
*   Calculates NDBI (Normalized Difference Built-up Index) and Albedo.
*   Displays interactive 2D (Folium) and 3D (PyDeck) maps.
*   Visualizes UHI Index as 3D column height.
*   Allows users to select map coloring based on UHI, Temperature, RH, NDBI, or Albedo.
*   Provides localized analysis for user-input coordinates using Google Gemini, explaining findings in simple terms and suggesting practical mitigation actions with reasoning.

**Requirements:**

*   Python 3.9+
*   Libraries: Streamlit, Pandas, GeoPandas, PyDeck, Folium, google-generativeai, pystac-client, planetary-computer, rioxarray, xarray, matplotlib.
*   Input CSV files  in the script directory.
*   Google Gemini API Key.

**Usage:**

1.  Install requirements (`pip install ...`).
2.  Place CSV files alongside the Python script.
3.  Run: `streamlit run your_script_name.py`
4.  Enter your Gemini API key in the sidebar.
5.  Select a date and click "Load/Refresh Data".
6.  Explore maps, changing color variables as needed.
7.  Enter coordinates and click "Get Analysis & Advice".
