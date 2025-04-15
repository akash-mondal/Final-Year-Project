import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
import pydeck as pdk
import google.generativeai as genai
import datetime
import warnings
import traceback
import folium
from streamlit_folium import st_folium
import os
import matplotlib.colors
import matplotlib.cm as cm

# Required imports for satellite data
import pystac_client
import planetary_computer as pc
import rioxarray
import xarray as xr

# --- Configuration ---
APP_TITLE = "UHI Analysis Tool"
NYC_CENTER_APPROX = [40.78, -73.96]
PYDECK_MAP_ZOOM = 11.5 # Slightly closer zoom
FOLIUM_MAP_ZOOM = 12
TARGET_CRS = "EPSG:4326"  # WGS 84 (Latitude/Longitude)
PROCESSING_CRS = "EPSG:32618" # UTM Zone 18N (for distance calculations)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_FILE = os.path.join(BASE_DIR, 'check.csv') # User's UHI data
WEATHER_FILE_MANHATTAN = os.path.join(BASE_DIR, 'manhattan.csv')
WEATHER_FILE_BRONX = os.path.join(BASE_DIR, 'bronx.csv')
WEATHER_STATIONS = {
    'Manhattan': {'file': WEATHER_FILE_MANHATTAN, 'coords': (-73.96, 40.78), 'desc': 'Central Park Area'},
    'Bronx': {'file': WEATHER_FILE_BRONX, 'coords': (-73.89, 40.86), 'desc': 'Bronx Area'}
}
BANDS_TO_LOAD = ["B02", "B03", "B04", "B08", "B11", "B12"]

# Visualization & Analysis Parameters
VIS_VARIABLES = ['UHI Index', 'Temperature (°C)', 'Relative Humidity (%)', 'NDBI', 'Albedo']
DEFAULT_COLOR_VAR = 'NDBI'
NEARBY_RADIUS_M = 150 # Radius for calculating nearby stats for AI context

# Pydeck Height Scaling
UHI_BASELINE_FOR_HEIGHT = 1.0
ELEVATION_SCALING_FACTOR = 1200 # Tunable height multiplier
COLUMN_RADIUS = 30

# --- Warnings ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='shapely')
warnings.filterwarnings('ignore', category=UserWarning, module='pyproj')
warnings.filterwarnings("ignore", message="The value of the smallest subnormal for <class 'numpy.float64'> type is zero.")

# --- Logging ---
if 'interaction_logs' not in st.session_state: st.session_state.interaction_logs = []
if 'data_load_logs' not in st.session_state: st.session_state.data_load_logs = []
def log_interaction(msg):
    timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]; log_entry = f"{timestamp} INTERACTION DEBUG: {msg}"; st.session_state.interaction_logs.append(log_entry); print(log_entry)
def log_data(msg):
    timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]; log_entry = f"{timestamp} DATA_LOAD DEBUG: {msg}"; st.session_state.data_load_logs.append(log_entry); print(log_entry)

# --- Gemini API ---
def configure_gemini(api_key):
    log_interaction("Attempting to configure Gemini API...")
    try: genai.configure(api_key=api_key); log_interaction("Gemini API configured successfully."); return True
    except Exception as e: st.error(f"Error configuring Gemini API: {e}"); log_interaction(f"Gemini API config error: {e}"); return False

def get_gemini_response(prompt):
    log_interaction("Sending prompt to Gemini...")
    print(f"--- GEMINI PROMPT START ---\n{prompt}\n--- GEMINI PROMPT END ---")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash'); response = model.generate_content(prompt)
        if not response.parts:
             safety_feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'No specific feedback.'; log_interaction(f"Gemini response blocked/empty. Feedback: {safety_feedback}")
             st.warning(f"Gemini response blocked/empty. Feedback: {safety_feedback}"); return f"Response blocked (Feedback: {safety_feedback})"
        log_interaction(f"Gemini response received (Length: {len(response.text)})."); return response.text
    except Exception as e: log_interaction(f"Error contacting Gemini: {e}"); st.error(f"Error contacting Gemini: {str(e)}"); return f"Error contacting Gemini: {str(e)}"

# --- Data Loading & Processing Functions ---

def parse_weather_datetime(df, filename):
    log_data(f"Attempting to parse datetime in {filename}...")
    try:
        df['datetime_str_cleaned'] = df['Date / Time'].str.replace(' EDT', '', regex=False)
        df['datetime_local'] = pd.to_datetime(df['datetime_str_cleaned'], format='%Y-%m-%d %H:%M:%S')
        df['datetime_utc'] = df['datetime_local'].dt.tz_localize('US/Eastern', ambiguous='infer').dt.tz_convert('UTC')
        log_data(f"Successfully parsed weather datetime for {filename}.")
        return df.drop(columns=['Date / Time', 'datetime_str_cleaned', 'datetime_local'])
    except Exception as e:
        problematic_value = "N/A"
        try: pd.to_datetime(df['Date / Time'].str.replace(' EDT', '', regex=False), format='%Y-%m-%d %H:%M:%S', errors='raise')
        except Exception as e_inner:
            if hasattr(e_inner, 'args') and len(e_inner.args) > 0: err_str = str(e_inner.args[0]); import re; match = re.search(r"time data '([^']*)'", err_str);
            if match: problematic_value = match.group(1)
        log_data(f"ERROR parsing datetime in {filename}: {e}. Problematic value example: '{problematic_value}'")
        return None

def parse_target_datetime(df, filename):
    log_data(f"Attempting to parse datetime in {filename}...")
    try:
        df['datetime_local'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
        df['datetime_utc'] = df['datetime_local'].dt.tz_localize('US/Eastern', ambiguous='infer').dt.tz_convert('UTC')
        log_data(f"Successfully parsed target datetime for {filename}.")
        return df.drop(columns=['datetime', 'datetime_local'])
    except Exception as e:
        problematic_value = "N/A"
        try: pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M', errors='raise')
        except Exception as e_inner:
             if hasattr(e_inner, 'args') and len(e_inner.args) > 0: err_str = str(e_inner.args[0]); import re; match = re.search(r"time data '([^']*)'", err_str);
             if match: problematic_value = match.group(1)
        log_data(f"ERROR parsing datetime in {filename}: {e}. Problematic value example: '{problematic_value}'")
        return None

@st.cache_data(ttl=3600)
def load_and_merge_csv_data():
    st.session_state.data_load_logs = []
    log_data("Starting CSV data load and merge...")
    gdf_target = None; gdf_merged = None
    try:
        df_target = pd.read_csv(TARGET_FILE)
        log_data(f"Loaded target file: {TARGET_FILE} ({len(df_target)} rows)")
        df_target = df_target.rename(columns={'Longitude': 'longitude', 'Latitude': 'latitude', 'UHI Index': 'uhi_index'})
        df_target_parsed = parse_target_datetime(df_target.copy(), os.path.basename(TARGET_FILE))
        if df_target_parsed is None: log_data("Stopping merge: target datetime parse failed."); st.error(f"Datetime parse error in {os.path.basename(TARGET_FILE)}"); return None
        geometry = [Point(xy) for xy in zip(df_target_parsed.longitude, df_target_parsed.latitude)]
        gdf_target = gpd.GeoDataFrame(df_target_parsed, geometry=geometry, crs=TARGET_CRS); log_data("Created GeoDataFrame from target data.")
        gdf_target = gdf_target.sort_values(by='datetime_utc'); log_data("Sorted target GeoDataFrame by datetime_utc.")
        gdf_target_proj = gdf_target.to_crs(PROCESSING_CRS); log_data(f"Projected target data to {PROCESSING_CRS}.")
    except FileNotFoundError: st.error(f"Error: Target file '{TARGET_FILE}' not found."); log_data(f"ERROR: Target file not found: {TARGET_FILE}"); return None
    except Exception as e: st.error(f"Error loading/processing target file {TARGET_FILE}: {e}"); log_data(f"ERROR loading target file: {e}\n{traceback.format_exc()}"); return None

    loaded_weather_dfs = {}
    for name, info in WEATHER_STATIONS.items():
        try:
            df_w = pd.read_csv(info['file']); log_data(f"Loaded weather file: {info['file']} ({len(df_w)} rows)")
            required_cols = ['Date / Time', 'Air Temp at Surface [degC]', 'Relative Humidity [percent]', 'Avg Wind Speed [m/s]', 'Solar Flux [W/m^2]']
            if not all(col in df_w.columns for col in required_cols): log_data(f"WARNING: Missing cols in {info['file']}. Skipping."); continue
            df_w = df_w[required_cols].copy(); df_w.columns = ['Date / Time', 'weather_temp', 'weather_rh', 'weather_wind', 'weather_solar']
            df_w_parsed = parse_weather_datetime(df_w.copy(), os.path.basename(info['file']))
            if df_w_parsed is None: st.warning(f"Datetime parse failed for {os.path.basename(info['file'])}."); continue
            df_w_parsed.set_index('datetime_utc', inplace=True)
            df_w_parsed.dropna(subset=['weather_temp', 'weather_rh'], inplace=True)
            loaded_weather_dfs[name] = df_w_parsed.sort_index(); log_data(f"Processed weather data for {name}.")
        except FileNotFoundError: st.warning(f"Weather file '{info['file']}' not found."); log_data(f"WARNING: Weather file not found: {info['file']}")
        except Exception as e: st.error(f"Error loading weather file {info['file']}: {e}"); log_data(f"ERROR loading weather file {info['file']}: {e}\n{traceback.format_exc()}")

    if not loaded_weather_dfs:
        log_data("WARNING: No weather data loaded. Merged data will lack weather info.")
        for col in ['weather_temp', 'weather_rh', 'weather_wind', 'weather_solar', 'nearest_station']:
            if col not in gdf_target.columns: gdf_target[col] = np.nan
        gdf_merged = gdf_target
    else:
        log_data("Assigning nearest weather station..."); station_points = {name: Point(info['coords'][0], info['coords'][1]) for name, info in WEATHER_STATIONS.items() if name in loaded_weather_dfs}
        station_gdf = gpd.GeoDataFrame({'station_name': list(station_points.keys())}, geometry=list(station_points.values()), crs=TARGET_CRS)
        station_gdf_proj = station_gdf.to_crs(PROCESSING_CRS)
        nearest_station_indices = gdf_target_proj.geometry.apply(lambda p: station_gdf_proj.distance(p).idxmin())
        gdf_target['nearest_station'] = nearest_station_indices.map(station_gdf_proj['station_name']); log_data("Nearest stations assigned.")
        log_data("Merging weather data..."); weather_data_list = []
        merge_failures = 0
        for index, target_point in gdf_target.iterrows():
            station_name = target_point['nearest_station']; target_dt = target_point['datetime_utc']
            point_weather = {'weather_temp': np.nan, 'weather_rh': np.nan, 'weather_wind': np.nan, 'weather_solar': np.nan}
            if pd.notna(target_dt) and station_name in loaded_weather_dfs:
                df_w_station = loaded_weather_dfs[station_name]; target_time_df = pd.DataFrame([target_dt], columns=['datetime_utc']).set_index('datetime_utc')
                try:
                    merged = pd.merge_asof(target_time_df, df_w_station, left_index=True, right_index=True, direction='nearest', tolerance=pd.Timedelta(minutes=15))
                    if merged.empty or merged[['weather_temp', 'weather_rh']].isnull().any(axis=1).iloc[0]: merge_failures += 1
                    else: nearest_weather = merged.iloc[0]; point_weather = {'weather_temp': nearest_weather['weather_temp'], 'weather_rh': nearest_weather['weather_rh'], 'weather_wind': nearest_weather.get('weather_wind', np.nan), 'weather_solar': nearest_weather.get('weather_solar', np.nan)}
                except Exception as merge_e: log_data(f"  Index {index}: ERROR during merge_asof: {merge_e}"); merge_failures += 1
            else: merge_failures += 1
            weather_data_list.append(point_weather)
        weather_df = pd.DataFrame(weather_data_list, index=gdf_target.index); gdf_merged = gdf_target.join(weather_df); log_data("Weather data merged.")
        if merge_failures > 0: log_data(f"WARNING: Failed to find matching weather data for {merge_failures}/{len(gdf_target)} points.")
        merged_nan_count = gdf_merged[['weather_temp', 'weather_rh']].isnull().sum().sum()
        if merged_nan_count > 0: log_data(f"INFO: {merged_nan_count} NaN values present in merged Temp/RH columns.")
    log_data("Finished load_and_merge_csv_data function.")
    return gdf_merged

@st.cache_data(ttl=3600)
def add_satellite_features(_gdf_data, selected_date_str):
    sat_logs = []
    def log_sat(msg): sat_logs.append(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} SAT_DEBUG: {msg}")
    log_sat(f"Starting satellite feature sampling for {selected_date_str} at {len(_gdf_data)} points...")
    if _gdf_data is None or _gdf_data.empty: log_sat("Input GDF empty, skipping satellite."); return _gdf_data, sat_logs
    gdf_processed = _gdf_data
    if _gdf_data.crs != TARGET_CRS:
        try: gdf_processed = _gdf_data.to_crs(TARGET_CRS); log_sat(f"Converted input GDF CRS from {_gdf_data.crs} to {TARGET_CRS}.")
        except Exception as e: log_sat(f"ERROR converting GDF to {TARGET_CRS}: {e}"); gdf_with_sat = _gdf_data.copy(); gdf_with_sat['NDBI'] = np.nan; gdf_with_sat['Albedo'] = np.nan; return gdf_with_sat, sat_logs
    bounds = gdf_processed.total_bounds; log_sat(f"Bounds for satellite search: {np.round(bounds, 4)}"); selected_date = datetime.datetime.strptime(selected_date_str, '%Y-%m-%d').date()
    gdf_with_sat = gdf_processed.copy()
    try:
        catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
        start_date = selected_date - datetime.timedelta(days=2); end_date = selected_date + datetime.timedelta(days=2); search_window = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        log_sat(f"Searching STAC API: bbox={np.round(bounds,4)}, datetime={search_window}")
        cloud_cover_limit = st.session_state.get('cloud_slider', 35); log_sat(f"Using max cloud cover limit: {cloud_cover_limit}%")
        search = catalog.search(collections=["sentinel-2-l2a"], bbox=bounds, datetime=search_window, query={"eo:cloud_cover": {"lt": cloud_cover_limit}})
        items = list(search.get_items()); log_sat(f"Found {len(items)} potential satellite items.")
        if not items: log_sat(f"No suitable satellite items found."); gdf_with_sat['NDBI'] = np.nan; gdf_with_sat['Albedo'] = np.nan; return gdf_with_sat, sat_logs
        items.sort(key=lambda item: (item.properties["eo:cloud_cover"], abs((datetime.datetime.fromisoformat(item.properties['datetime'].replace("Z", "+00:00")).date() - selected_date).days)))
        selected_item = items[0]; item_datetime = datetime.datetime.fromisoformat(selected_item.properties['datetime'].replace("Z", "+00:00"))
        log_sat(f"Selected satellite item: {selected_item.id} ({item_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}, Cloud: {selected_item.properties['eo:cloud_cover']:.1f}%)")
        band_data = {}; raster_crs = None; gdf_points_proj = None; all_bands_sampled = True
        for band in BANDS_TO_LOAD:
             try:
                asset = selected_item.assets.get(band)
                if not asset: log_sat(f"Asset {band} not found."); band_data[band] = pd.Series(np.nan, index=gdf_with_sat.index); all_bands_sampled = False; continue
                href = pc.sign(asset.href)
                try:
                     with rioxarray.open_rasterio(href, chunks={'x': 1024, 'y': 1024}) as da:
                        if 'band' in da.dims: da = da.squeeze('band', drop=True); da = da.rio.write_crs(da.rio.crs)
                        if raster_crs is None:
                            raster_crs = da.rio.crs; log_sat(f"Satellite raster CRS: {raster_crs}")
                            if gdf_with_sat.crs != raster_crs:
                                # --- CORRECTED BLOCK ---
                                try:
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore", category=UserWarning) # Statement 1 inside with
                                        gdf_points_proj = gdf_with_sat.to_crs(raster_crs)    # Statement 2 inside with
                                    # Log outside the warnings context, after successful projection
                                    log_sat(f"Projected CSV points from {gdf_with_sat.crs} to raster CRS {raster_crs} for sampling.")
                                except Exception as crs_e:
                                    log_sat(f"ERROR projecting CSV points to raster CRS {raster_crs}: {crs_e}") # Log specific CRS
                                    all_bands_sampled = False
                                    break # Critical error, stop processing bands
                                # --- END CORRECTED BLOCK ---
                            else:
                                gdf_points_proj = gdf_with_sat
                                log_sat("CSV points already in raster CRS.")
                        if gdf_points_proj is None: log_sat("Projected GDF is None, skipping band."); all_bands_sampled = False; band_data[band] = pd.Series(np.nan, index=gdf_with_sat.index); continue
                        x_coords = xr.DataArray(gdf_points_proj.geometry.x.values, dims="points"); y_coords = xr.DataArray(gdf_points_proj.geometry.y.values, dims="points")
                        log_sat(f"Sampling satellite band {band}..."); sampled_values = da.sel(x=x_coords, y=y_coords, method="nearest").compute().values; band_data[band] = pd.Series(sampled_values, index=gdf_with_sat.index)
                        log_sat(f"Sampled {band} ({pd.isna(sampled_values).sum()}/{len(gdf_with_sat)} points were NaN).")
                except ImportError as imp_err:
                    if 'dask' in str(imp_err): st.error("CRITICAL: 'dask' needed."); log_sat("CRITICAL: 'dask' not found."); all_bands_sampled = False; break
                    else: log_sat(f"Import error sampling {band}: {imp_err}"); all_bands_sampled = False; band_data[band] = pd.Series(np.nan, index=gdf_with_sat.index)
                except Exception as rio_open_err: log_sat(f"Raster error sampling {band}: {rio_open_err}"); all_bands_sampled = False; band_data[band] = pd.Series(np.nan, index=gdf_with_sat.index); continue
             except Exception as e: log_sat(f"General error sampling {band}: {e}"); all_bands_sampled = False; band_data[band] = pd.Series(np.nan, index=gdf_with_sat.index)
        if all_bands_sampled or band_data:
            sampled_df = pd.DataFrame(band_data, index=gdf_with_sat.index); log_sat("Calculating NDBI and Albedo...")
            scale = 10000.0
            b02 = sampled_df.get("B02", pd.Series(np.nan, index=gdf_with_sat.index)).astype(float); b04 = sampled_df.get("B04", pd.Series(np.nan, index=gdf_with_sat.index)).astype(float); b08 = sampled_df.get("B08", pd.Series(np.nan, index=gdf_with_sat.index)).astype(float); b11 = sampled_df.get("B11", pd.Series(np.nan, index=gdf_with_sat.index)).astype(float); b12 = sampled_df.get("B12", pd.Series(np.nan, index=gdf_with_sat.index)).astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                ndbi = (b11 - b08) / (b11 + b08)
                albedo = (0.356*(b02/scale) + 0.130*(b04/scale) + 0.373*(b08/scale) + 0.085*(b11/scale) + 0.072*(b12/scale) - 0.0018); albedo = np.clip(albedo, 0, 1)
            gdf_with_sat['NDBI'] = ndbi; gdf_with_sat['Albedo'] = albedo
            log_sat(f"Added NDBI/Albedo columns. NDBI NaNs: {gdf_with_sat['NDBI'].isnull().sum()}, Albedo NaNs: {gdf_with_sat['Albedo'].isnull().sum()}")
            if not all_bands_sampled: log_sat("WARNING: Sampling issues occurred for one or more bands.")
        else: log_sat("Skipping NDBI/Albedo calculation."); gdf_with_sat['NDBI'] = np.nan; gdf_with_sat['Albedo'] = np.nan
    except pystac_client.exceptions.APIError as api_err: log_sat(f"PC API error: {api_err}"); st.error(f"STAC API Error: {api_err}"); gdf_with_sat['NDBI'] = np.nan; gdf_with_sat['Albedo'] = np.nan
    except ImportError as imp_err: log_sat(f"Import error: {imp_err}"); st.error(f"Import Error: {imp_err}"); gdf_with_sat['NDBI'] = np.nan; gdf_with_sat['Albedo'] = np.nan
    except Exception as e: log_sat(f"Unexpected error: {e}\n{traceback.format_exc()}"); st.error(f"Error during satellite processing: {e}"); gdf_with_sat['NDBI'] = np.nan; gdf_with_sat['Albedo'] = np.nan
    log_sat("Finished adding satellite features."); return gdf_with_sat, sat_logs

# --- Helper Function for AI Context ---
def get_context_for_gemini(map_data, lat, lon, radius_m=NEARBY_RADIUS_M):
    """Gathers data for the nearest point and summarizes nearby points."""
    nearest_point_str = "Nearest Point: No data loaded or point not found."
    nearby_summary_str = f"Nearby Area (within {radius_m}m): No nearby points found or data unavailable."
    nearest_station_name = "N/A"

    if map_data is None or map_data.empty:
        return nearest_point_str, nearby_summary_str, nearest_station_name

    try:
        entered_point = Point(lon, lat)
        map_data_for_analysis = map_data.copy()

        # Ensure CRS
        if map_data_for_analysis.crs is None: map_data_for_analysis.set_crs(TARGET_CRS, inplace=True)
        elif map_data_for_analysis.crs != TARGET_CRS: map_data_for_analysis = map_data_for_analysis.to_crs(TARGET_CRS)

        # Project for distance/buffer
        map_data_proj = map_data_for_analysis.to_crs(PROCESSING_CRS)
        entered_point_gdf = gpd.GeoDataFrame([1], geometry=[entered_point], crs=TARGET_CRS)
        entered_point_proj = entered_point_gdf.to_crs(PROCESSING_CRS).geometry.iloc[0]

        # --- Nearest Point ---
        distances_m = map_data_proj.geometry.distance(entered_point_proj)
        if distances_m.empty: # Handle case where no points exist after projection? (unlikely but safe)
             return nearest_point_str, nearby_summary_str, nearest_station_name
        nearest_index = distances_m.idxmin()
        nearest_point_data = map_data_for_analysis.loc[nearest_index]
        min_distance_m = distances_m.min()
        nearest_station_name = nearest_point_data.get('nearest_station', 'N/A')

        # Format nearest point data
        np_ndbi = nearest_point_data.get('NDBI', np.nan); np_albedo = nearest_point_data.get('Albedo', np.nan)
        np_uhi = nearest_point_data.get('uhi_index', np.nan); np_temp = nearest_point_data.get('weather_temp', np.nan)
        np_rh = nearest_point_data.get('weather_rh', np.nan); np_dt_utc = nearest_point_data.get('datetime_utc', pd.NaT)
        np_dt_str = np_dt_utc.strftime('%Y-%m-%d %H:%M UTC') if pd.notna(np_dt_utc) else 'N/A'

        nearest_point_str = (
            f"Nearest Point (approx {min_distance_m:.0f}m away, recorded around {np_dt_str}):\n"
            f"- UHI Index: {np_uhi:.3f}\n" if pd.notna(np_uhi) else "- UHI Index: N/A\n"
            f"- Air Temperature: {np_temp:.1f}°C\n" if pd.notna(np_temp) else "- Air Temperature: N/A\n"
            f"- Relative Humidity: {np_rh:.1f}%\n" if pd.notna(np_rh) else "- Relative Humidity: N/A\n"
            f"- NDBI (Satellite): {np_ndbi:.3f}\n" if pd.notna(np_ndbi) else "- NDBI (Satellite): N/A\n"
            f"- Albedo (Satellite): {np_albedo:.3f}" if pd.notna(np_albedo) else "- Albedo (Satellite): N/A"
        )

        # --- Nearby Points Summary ---
        nearby_indices = map_data_proj.geometry.within(entered_point_proj.buffer(radius_m))
        nearby_points = map_data_for_analysis[nearby_indices]

        if not nearby_points.empty and len(nearby_points) > 1:
             cols_to_summarize = {'uhi_index': '.3f', 'weather_temp': '.1f', 'weather_rh': '.1f', 'NDBI': '.3f', 'Albedo': '.3f'}
             summary_lines = [f"Nearby Area Summary ({len(nearby_points)} points within {radius_m}m):"]
             for col, fmt in cols_to_summarize.items():
                 if col in nearby_points.columns:
                     valid_data = nearby_points[col].dropna()
                     if not valid_data.empty:
                         mean_val = valid_data.mean(); min_val = valid_data.min(); max_val = valid_data.max()
                         summary_lines.append(f"- {col}: Avg={mean_val:{fmt}}, Min={min_val:{fmt}}, Max={max_val:{fmt}}")
                     else: summary_lines.append(f"- {col}: No valid data nearby")
                 else: summary_lines.append(f"- {col}: Data not available")
             nearby_summary_str = "\n".join(summary_lines)
        elif len(nearby_points) == 1: nearby_summary_str = f"Nearby Area (within {radius_m}m): Only the nearest point was found in this radius."
        else: nearby_summary_str = f"Nearby Area (within {radius_m}m): No measurement points found in this radius." # Explicitly state if none found

    except Exception as e:
        log_interaction(f"Error getting AI context: {e}")
        nearest_point_str = f"Nearest Point: Error processing data - {e}"
        nearby_summary_str = f"Nearby Area (within {radius_m}m): Error processing data - {e}"

    return nearest_point_str, nearby_summary_str, nearest_station_name

# --- Helper Function for Color Mapping ---
def get_color_for_variable(value, variable_name, v_min, v_max, output_format='rgba'):
    """Maps a variable's value to a color based on predefined schemes."""
    cmap_temp = cm.get_cmap('coolwarm'); cmap_rh = cm.get_cmap('YlGnBu'); cmap_ndbi = cm.get_cmap('RdYlGn_r')
    cmap_albedo = cm.get_cmap('Greys'); cmap_uhi = cm.get_cmap('YlOrRd')

    invert_norm = False
    if variable_name == 'Temperature (°C)': cmap = cmap_temp
    elif variable_name == 'Relative Humidity (%)': cmap = cmap_rh
    elif variable_name == 'NDBI': cmap = cmap_ndbi
    elif variable_name == 'Albedo': cmap = cmap_albedo
    elif variable_name == 'UHI Index': cmap = cmap_uhi
    else: cmap = cm.get_cmap('viridis')

    if pd.isna(value) or pd.isna(v_min) or pd.isna(v_max) or v_min >= v_max:
        if output_format == 'rgba': return [128, 128, 128, 150] # Grey RGBA
        else: return '#808080' # Grey Hex

    norm_value = (value - v_min) / (v_max - v_min); norm_value = max(0.0, min(1.0, norm_value))
    if invert_norm: norm_value = 1.0 - norm_value
    color = cmap(norm_value)

    if output_format == 'rgba': return [int(c * 255) for c in color[:3]] + [200]
    else: return matplotlib.colors.to_hex(color)

# --- Streamlit App Layout ---
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.markdown("Select date, load data, choose map variables, view maps, **enter coordinates below**, then click **'Get Analysis & Advice'**.")
log_interaction("App script started/restarted.")

# --- Initialize Session State ---
if "gemini_configured" not in st.session_state: st.session_state.gemini_configured = False
if "map_data" not in st.session_state: st.session_state.map_data = None
if "load_attempted" not in st.session_state: st.session_state.load_attempted = False
if "mitigation_advice" not in st.session_state: st.session_state.mitigation_advice = None
if "data_load_logs" not in st.session_state: st.session_state.data_load_logs = []
if 'interaction_logs' not in st.session_state: st.session_state.interaction_logs = []
if 'vis_ranges' not in st.session_state: st.session_state.vis_ranges = {}
if 'color_variable_3d' not in st.session_state: st.session_state.color_variable_3d = DEFAULT_COLOR_VAR
if 'color_variable_2d' not in st.session_state: st.session_state.color_variable_2d = DEFAULT_COLOR_VAR

# --- Map Column Name Mapping ---
COLUMN_MAPPING = {'UHI Index': 'uhi_index', 'Temperature (°C)': 'weather_temp', 'Relative Humidity (%)': 'weather_rh', 'NDBI': 'NDBI', 'Albedo': 'Albedo'}

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google Gemini API Key:", type="password", key="api_key_input")
    if api_key and not st.session_state.gemini_configured:
        st.session_state.gemini_configured = configure_gemini(api_key)
        if st.session_state.gemini_configured: st.success("Gemini API Configured!")
        else: st.error("Invalid API Key or configuration error.")

    st.header("Data Source & Date")
    st.info(f"Target: {os.path.basename(TARGET_FILE)}\nWeather: Manhattan, Bronx")
    today = datetime.date.today(); min_date = datetime.date(2017, 3, 28)
    selected_date = st.date_input("Select Date (for Satellite Image):", datetime.date(2021, 7, 24), min_value=min_date, max_value=today, key="date_selector")
    st.slider("Max Cloud Cover (%):", 0, 100, 35, key="cloud_slider", help="Max cloud cover for satellite image search.")

    if st.button("Load/Refresh Data", key="load_data_button", type="primary", use_container_width=True):
        log_interaction("'Load/Refresh Data' button clicked.")
        st.session_state.load_attempted = True
        st.session_state.map_data = None; st.session_state.mitigation_advice = None; st.session_state.vis_ranges = {}
        st.session_state.data_load_logs = []
        merged_gdf = None; final_gdf = None; logs_csv = []; logs_sat = []
        with st.spinner("Loading CSV, merging weather, sampling satellite..."):
            log_interaction("Starting CSV load and merge..."); merged_gdf = load_and_merge_csv_data()
            logs_csv = st.session_state.get('data_load_logs', []).copy()
            if merged_gdf is not None and not merged_gdf.empty:
                 log_interaction("Starting satellite sampling..."); final_gdf, logs_sat_func = add_satellite_features(merged_gdf, selected_date.strftime('%Y-%m-%d'))
                 st.session_state.map_data = final_gdf; logs_sat = logs_sat_func
                 st.session_state.vis_ranges = {} # Reset ranges before calculating
                 for display_name, col_name in COLUMN_MAPPING.items():
                     if col_name in final_gdf.columns:
                         valid_data = final_gdf[col_name].dropna()
                         if not valid_data.empty:
                             st.session_state.vis_ranges[display_name] = {'min': valid_data.min(), 'max': valid_data.max(), 'col': col_name}
                             log_data(f"Vis range for {display_name}: min={valid_data.min():.2f}, max={valid_data.max():.2f}")
                         else: log_data(f"No valid data for {display_name} to calculate range.")
                     else: log_data(f"Column {col_name} not found for vis range calculation.")
                 # Ensure default color vars are valid after load
                 if DEFAULT_COLOR_VAR not in st.session_state.vis_ranges:
                     first_valid_var = next(iter(st.session_state.vis_ranges), None)
                     st.session_state.color_variable_3d = first_valid_var
                     st.session_state.color_variable_2d = first_valid_var
                 else:
                      st.session_state.color_variable_3d = DEFAULT_COLOR_VAR
                      st.session_state.color_variable_2d = DEFAULT_COLOR_VAR

            else: log_interaction("Base data load failed."); st.error("Failed to load/merge base data."); st.session_state.map_data = None
            st.session_state.data_load_logs = logs_csv + logs_sat
            log_interaction("Data loading process finished.")
        st.rerun()

    st.markdown("---")
    st.subheader("Map Legend")
    st.markdown("*(Legend shows range for selected Color Variable)*")
    st.markdown("**3D Map Columns:**")
    st.markdown(f"- **Height:** UHI Index (scaled x{ELEVATION_SCALING_FACTOR})")
    # Display legend for 3D color variable
    color_var_3d_sb = st.session_state.get('color_variable_3d', None)
    if color_var_3d_sb and color_var_3d_sb in st.session_state.vis_ranges:
        range_3d = st.session_state.vis_ranges[color_var_3d_sb]
        st.markdown(f"- **Color:** {color_var_3d_sb} ({range_3d['min']:.2f} to {range_3d['max']:.2f})")
    else: st.markdown("- **Color:** *(Select variable on map)*")

    st.markdown("**2D Map Circles:**")
    # Display legend for 2D color variable
    color_var_2d_sb = st.session_state.get('color_variable_2d', None)
    if color_var_2d_sb and color_var_2d_sb in st.session_state.vis_ranges:
        range_2d = st.session_state.vis_ranges[color_var_2d_sb]
        st.markdown(f"- **Color:** {color_var_2d_sb} ({range_2d['min']:.2f} to {range_2d['max']:.2f})")
    else: st.markdown("- **Color:** *(Select variable on map)*")

    st.markdown("---")
    st.caption("Data: Local CSVs + Sentinel-2 via MS Planetary Computer.")
# End sidebar

# Display Data Loading Logs
if st.session_state.data_load_logs:
    expand_logs = st.session_state.load_attempted and st.session_state.map_data is None
    with st.expander("Show Data Loading Logs", expanded=expand_logs): st.text("\n".join(st.session_state.data_load_logs))

# --- Main Area Layout (Maps and Controls) ---
col1, col2 = st.columns([0.65, 0.35], gap="medium")

map_data = st.session_state.get('map_data', None)
vis_ranges = st.session_state.get('vis_ranges', {})
available_vis_vars = list(vis_ranges.keys()) # Get available variables AFTER data load attempt

with col1:
    # --- 3D Map Section ---
    st.subheader("3D Map Viewer (PyDeck)")
    # Ensure options are valid before setting selectbox
    idx_3d = available_vis_vars.index(st.session_state.color_variable_3d) if st.session_state.color_variable_3d in available_vis_vars else 0
    color_var_3d = st.selectbox("Select variable for **Column Color**:", options=available_vis_vars, key='sb_color_variable_3d', index=idx_3d)
    # Update session state if changed
    if color_var_3d != st.session_state.color_variable_3d:
        st.session_state.color_variable_3d = color_var_3d
        st.rerun() # Rerun to update map with new color

    if map_data is not None and not map_data.empty and color_var_3d and color_var_3d in vis_ranges: # Check if var exists in ranges
        try:
            pydeck_data = map_data.copy()
            pydeck_data['latitude'] = pydeck_data.geometry.y; pydeck_data['longitude'] = pydeck_data.geometry.x
            pydeck_data['uhi_for_elevation'] = pd.to_numeric(pydeck_data['uhi_index'], errors='coerce').fillna(UHI_BASELINE_FOR_HEIGHT)

            color_col_info = vis_ranges[color_var_3d]; color_col_name = color_col_info['col']; v_min = color_col_info['min']; v_max = color_col_info['max']
            pydeck_data['color'] = pydeck_data[color_col_name].apply(get_color_for_variable, args=(color_var_3d, v_min, v_max, 'rgba'))

            pydeck_data['NDBI_str'] = pydeck_data['NDBI'].map('{:.3f}'.format).fillna('N/A') if 'NDBI' in pydeck_data else 'N/A'
            pydeck_data['Albedo_str'] = pydeck_data['Albedo'].map('{:.3f}'.format).fillna('N/A') if 'Albedo' in pydeck_data else 'N/A'
            pydeck_data['weather_temp_str'] = pydeck_data['weather_temp'].map('{:.1f}'.format).fillna('N/A') if 'weather_temp' in pydeck_data else 'N/A'
            pydeck_data['weather_rh_str'] = pydeck_data['weather_rh'].map('{:.1f}'.format).fillna('N/A') if 'weather_rh' in pydeck_data else 'N/A'
            pydeck_data['uhi_index_str'] = pydeck_data['uhi_index'].map('{:.3f}'.format).fillna('N/A') if 'uhi_index' in pydeck_data else 'N/A'

            PYDECK_TOOLTIP = {
                "html": """
                    <b>Coords:</b> {latitude:.4f}, {longitude:.4f}<br/>
                    <b>UHI Index:</b> {uhi_index_str}<br/>
                    <b>Temp (°C):</b> {weather_temp_str}<br/>
                    <b>RH (%):</b> {weather_rh_str}<br/>
                    <b>NDBI:</b> {NDBI_str}<br/>
                    <b>Albedo:</b> {Albedo_str}<br/>
                    <hr style='margin: 2px 0;'>
                    <i>Height ~ UHI Index (Raw: {uhi_for_elevation:.3f})</i>
                    """,
                "style": {"backgroundColor": "darkslategray", "color": "white", "font-size": "11px"}
            }
            required_pydeck_cols = ['longitude', 'latitude', 'color','uhi_for_elevation', 'uhi_index_str', 'weather_temp_str', 'weather_rh_str', 'NDBI_str', 'Albedo_str']

            if all(col in pydeck_data.columns for col in required_pydeck_cols):
                pydeck_view_state = pdk.ViewState(latitude=NYC_CENTER_APPROX[0], longitude=NYC_CENTER_APPROX[1], zoom=PYDECK_MAP_ZOOM, pitch=55, bearing=-15)
                pydeck_layer = pdk.Layer("ColumnLayer", data=pd.DataFrame(pydeck_data[required_pydeck_cols]), get_position=["longitude", "latitude"], get_elevation='uhi_for_elevation', elevation_scale=ELEVATION_SCALING_FACTOR, radius=COLUMN_RADIUS, get_fill_color="color", pickable=True, auto_highlight=True, extruded=True)
                pydeck_deck = pdk.Deck(layers=[pydeck_layer], initial_view_state=pydeck_view_state, map_style="mapbox://styles/mapbox/light-v10", tooltip=PYDECK_TOOLTIP)
                st.pydeck_chart(pydeck_deck, use_container_width=True)
            else: missing_cols = [c for c in required_pydeck_cols if c not in pydeck_data.columns]; st.warning(f"3D map missing columns: {missing_cols}"); log_interaction(f"PyDeck missing columns: {missing_cols}")
        except Exception as pydeck_err: st.error(f"Error rendering 3D map: {pydeck_err}"); log_interaction(f"PyDeck render failed: {pydeck_err}\n{traceback.format_exc()}")
    elif st.session_state.load_attempted: st.info("Load data to view 3D map.")
    else: st.info("Click 'Load/Refresh Data' to begin.")

    st.markdown("---")

    # --- 2D Map Section ---
    st.subheader("2D Map Viewer (Folium)")
    idx_2d = available_vis_vars.index(st.session_state.color_variable_2d) if st.session_state.color_variable_2d in available_vis_vars else 0
    color_var_2d = st.selectbox("Select variable for **Circle Color**:", options=available_vis_vars, key='sb_color_variable_2d', index=idx_2d)
    if color_var_2d != st.session_state.color_variable_2d:
        st.session_state.color_variable_2d = color_var_2d
        st.rerun()

    if map_data is not None and not map_data.empty and color_var_2d and color_var_2d in vis_ranges:
        try:
            m = folium.Map(location=NYC_CENTER_APPROX, zoom_start=FOLIUM_MAP_ZOOM, tiles="CartoDB positron")
            folium_data = map_data.copy()
            color_col_info_2d = vis_ranges[color_var_2d]; color_col_name_2d = color_col_info_2d['col']; v_min_2d = color_col_info_2d['min']; v_max_2d = color_col_info_2d['max']
            folium_data['hex_color'] = folium_data[color_col_name_2d].apply(get_color_for_variable, args=(color_var_2d, v_min_2d, v_max_2d, 'hex'))

            for idx, row in folium_data.iterrows():
                lat = row.geometry.y; lon = row.geometry.x; color = row['hex_color']
                uhi_val = row.get('uhi_index', np.nan); temp_val = row.get('weather_temp', np.nan); rh_val = row.get('weather_rh', np.nan); ndbi_val = row.get('NDBI', np.nan); albedo_val = row.get('Albedo', np.nan)
                uhi_str = f"{uhi_val:.3f}" if pd.notna(uhi_val) else "N/A"; temp_str = f"{temp_val:.1f}°C" if pd.notna(temp_val) else "N/A"; rh_str = f"{rh_val:.1f}%" if pd.notna(rh_val) else "N/A"; ndbi_str = f"{ndbi_val:.3f}" if pd.notna(ndbi_val) else "N/A"; albedo_str = f"{albedo_val:.3f}" if pd.notna(albedo_val) else "N/A"
                tooltip_html = f"<b>Coords:</b> {lat:.4f}, {lon:.4f}<br><b>UHI:</b> {uhi_str}<br><b>Temp:</b> {temp_str}<br><b>RH:</b> {rh_str}<br><b>NDBI:</b> {ndbi_str}<br><b>Albedo:</b> {albedo_str}"
                folium.CircleMarker(location=[lat, lon], radius=4, color=color, fill=True, fill_color=color, fill_opacity=0.7, tooltip=tooltip_html).add_to(m)
            st_folium(m, use_container_width=True, height=450, key="folium_map_display_only")
        except Exception as map_render_error: st.error(f"Error rendering 2D map: {map_render_error}"); log_interaction(f"Folium render error: {map_render_error}\n{traceback.format_exc()}")
    elif st.session_state.load_attempted: st.info("Load data to view 2D map.")
# End col1

with col2:
    # --- Coordinate Input and Analysis ---
    st.subheader("Location Analysis & Advice")
    log_interaction("Setting up coordinate input section.")
    default_lat = NYC_CENTER_APPROX[0]; default_lon = NYC_CENTER_APPROX[1]
    lat_input = st.number_input("Enter Latitude:", min_value=40.4, max_value=41.0, value=default_lat, step=0.0001, format="%.4f", key="lat_coord_input")
    lon_input = st.number_input("Enter Longitude:", min_value=-74.3, max_value=-73.7, value=default_lon, step=0.0001, format="%.4f", key="lon_coord_input")
    advice_button_disabled = not st.session_state.gemini_configured or map_data is None or map_data.empty

    if st.button("Get Analysis & Advice", disabled=advice_button_disabled, key="get_advice_button", type="primary", use_container_width=True):
        log_interaction("'Get Analysis & Advice' button clicked.")
        if map_data is not None and not map_data.empty:
            with st.spinner("Analyzing location and asking assistant..."):
                try:
                    log_interaction(f"Analyzing entered coords: Lat={lat_input:.4f}, Lon={lon_input:.4f}")
                    nearest_point_str, nearby_summary_str, nearest_station_name = get_context_for_gemini(map_data, lat_input, lon_input)
                    station_desc = WEATHER_STATIONS.get(nearest_station_name, {}).get('desc', 'Unknown Area')

                    # Construct the NEW prompt for Gemini
                    full_prompt = f"""You are an AI assistant providing clear, easy-to-understand advice on urban heat island (UHI) effects and mitigation for residents or local planners in NYC. Your language should be simple and avoid technical terms where possible.

Location Context:
User Coordinates: Lat={lat_input:.4f}, Lon={lon_input:.4f}
Nearest Weather Station Data From: {nearest_station_name} ({station_desc})

Data Summary:
{nearest_point_str}
{nearby_summary_str}

Your Task:
Based *only* on the provided data summary:

1.  **Explain the Findings (Simple Language):** Describe the heat situation near the requested coordinates in plain terms.
    *   Instead of just stating the UHI index, explain if the location tends to be hotter than average rural surroundings and roughly by how much (e.g., "This spot seems significantly warmer than areas outside the city." or "This location is moderately warmer...").
    *   Explain what the temperature and humidity levels *feel* like or imply for comfort/risk (e.g., "The reported temperature is quite high, and combined with the humidity, it would likely feel very uncomfortable and potentially dangerous during peak sun.").
    *   Explain what the satellite data suggests about the physical environment (e.g., "Satellite views suggest this area has many buildings and paved surfaces, with less green space like parks or trees." or "Satellite views indicate a mix of buildings and some vegetation."). Do *not* use the terms "NDBI" or "Index".
    *   Explain what the satellite data suggests about the surfaces (e.g., "The surfaces in this area appear mostly dark, meaning they likely absorb a lot of sun heat." or "There's a mix of darker and lighter surfaces."). Do *not* use the term "Albedo" or "Index".
    *   Briefly comment on whether the immediate surroundings (nearby area summary) show similar or different conditions, indicating if the specific point is typical for its neighborhood (e.g., "The surrounding area generally shares these characteristics." or "Conditions vary somewhat nearby, with some spots showing [different characteristic].").
    *   Acknowledge if data is from a point somewhat distant (>100m) from the user's precise coordinates (e.g., "Note: The closest measurement was taken about [X] meters away, so conditions at your exact spot might differ slightly.").

2.  **Provide Actionable Mitigation Advice (with Reasoning):** Offer 2-4 practical, localized suggestions relevant to the findings.
    *   For each suggestion, briefly explain *why* it helps reduce local heat in simple terms (e.g., "Planting trees helps because their leaves provide shade, cooling the ground and the air around them." or "Using lighter-colored materials for roofs and pavements helps because they reflect sunlight away instead of absorbing it as heat.").
    *   Tailor suggestions to the findings (e.g., if the area seems built-up with dark surfaces, focus on greening, cool roofs/pavements. If humidity is high, mention ventilation or reducing waste heat.).
    *   Keep advice realistic for an NYC setting (mentioning things like street trees, green roofs, community gardens, cool pavement coatings, choosing lighter paint colors where possible).

Structure your response clearly with headings like "Heat Situation Explained" and "Mitigation Suggestions". Use bullet points for clarity.
"""
                    response_text = get_gemini_response(full_prompt)
                    st.session_state.mitigation_advice = response_text
                    log_interaction("Mitigation advice generated and stored in session state.")
                    st.rerun()
                except Exception as advice_err: st.error(f"Error getting advice: {advice_err}"); log_interaction(f"Error during advice: {advice_err}\n{traceback.format_exc()}"); st.session_state.mitigation_advice = "Error retrieving advice."
        else: log_interaction("Get advice clicked but map_data not available."); st.warning("Map data is not loaded.")

    # Display status messages or advice
    if not st.session_state.gemini_configured: st.warning("Enter Google Gemini API Key in sidebar to enable analysis.")
    elif map_data is None or map_data.empty:
        if st.session_state.load_attempted: st.warning("Data loading failed or is empty. Cannot get advice.")
        else: st.info("Load data via sidebar to enable analysis.")

    if st.session_state.mitigation_advice:
         st.markdown("---"); log_interaction("Displaying stored mitigation advice.")
         st.subheader("Analysis & Mitigation Advice")
         with st.container(border=True): st.markdown(st.session_state.mitigation_advice)

    if st.session_state.interaction_logs:
        st.markdown("---")
        with st.expander("Show Interaction Logs"): st.text("\n".join(st.session_state.interaction_logs[::-1]))
# End col2

log_interaction("App script run finished.")
