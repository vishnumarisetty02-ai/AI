# ============================================================
# GEOVISION AI
# AI-BASED URBAN PARCEL MAPPING + CADASTRAL FEATURE EXTRACTION
# + DRONE LAND SURVEY + GEO-TIFF + GIS PROCESSING
#
# SINGLE FILE STREAMLIT APPLICATION
# ============================================================

import os
import io
import json
import math
import time
import random
import shutil
import zipfile
import tempfile
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.validation import explain_validity

# ------------------------------------------------------------
# Optional GIS raster support
# ------------------------------------------------------------
try:
    import rasterio
    from rasterio.io import MemoryFile
    RASTERIO_AVAILABLE = bool(rasterio)
except ImportError:
    rasterio = None
    MemoryFile = None
    RASTERIO_AVAILABLE = False

# ------------------------------------------------------------
# PDF support
# ------------------------------------------------------------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle
    )
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ------------------------------------------------------------
# Map support
# ------------------------------------------------------------
try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except ImportError:
    PYDECK_AVAILABLE = False

try:
    import folium
    from folium.plugins import Draw
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GeoVision AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #0f4c81;
    }

    .subtitle {
        color: #64748b;
        font-size: 1rem;
    }

    .metric-card {
        background: linear-gradient(
            135deg,
            #0f172a,
            #123c63
        );
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #cbd5e1;
    }

    .high {
        background: #dcfce7;
        color: #166534;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }

    .inspect {
        background: #fef3c7;
        color: #92400e;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }

    .field {
        background: #fee2e2;
        color: #991b1b;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }

    .notice {
        background: #fff7d6;
        border: 1px solid #eab308;
        padding: 15px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


def render_animated_globe():

    components.html(
        """
        <!doctype html>
        <html>
        <head>
        <style>
            * { box-sizing: border-box; }
            html, body {
                width: 100%;
                height: 100%;
                margin: 0;
                overflow: hidden;
                background: transparent;
                font-family: 'Segoe UI', sans-serif;
            }
            .hero {
                position: relative;
                height: 276px;
                overflow: hidden;
                border: 1px solid rgba(15, 76, 129, .16);
                border-radius: 10px;
                background:
                    radial-gradient(ellipse at 76% 50%, rgba(31, 163, 160, .18), transparent 37%),
                    linear-gradient(112deg, #f5fbfa 0%, #e9f5f2 54%, #e1f1ef 100%);
            }
            .hero::before {
                position: absolute;
                inset: 0;
                background-image:
                    linear-gradient(rgba(17, 90, 108, .045) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(17, 90, 108, .045) 1px, transparent 1px);
                background-size: 32px 32px;
                content: '';
                mask-image: linear-gradient(90deg, transparent, black);
            }
            .copy {
                position: absolute;
                z-index: 2;
                top: 50%;
                left: clamp(22px, 5vw, 64px);
                max-width: 43%;
                transform: translateY(-50%);
            }
            .eyebrow {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 13px;
                color: #147c79;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1.2px;
            }
            .pulse {
                width: 7px;
                height: 7px;
                border-radius: 50%;
                background: #20a477;
                box-shadow: 0 0 0 4px rgba(32, 164, 119, .12);
            }
            h2 {
                margin: 0;
                color: #123744;
                font-size: clamp(23px, 3.2vw, 35px);
                line-height: 1.12;
                font-weight: 750;
            }
            .description {
                max-width: 360px;
                margin: 12px 0 0;
                color: #526b70;
                font-size: 13px;
                line-height: 1.55;
            }
            .coordinates {
                margin-top: 18px;
                color: #147c79;
                font: 10px/1.5 'Consolas', monospace;
            }
            #scene {
                position: absolute;
                inset: 0 0 0 44%;
            }
            #scene canvas { display: block; width: 100%; height: 100%; }
            .fallback-globe {
                position: absolute;
                top: 50%;
                left: 55%;
                width: 210px;
                aspect-ratio: 1;
                border: 1px solid rgba(61, 195, 184, .7);
                border-radius: 50%;
                background:
                    repeating-linear-gradient(0deg, transparent 0 25px, rgba(114, 221, 199, .3) 26px 27px),
                    repeating-linear-gradient(90deg, transparent 0 25px, rgba(114, 221, 199, .3) 26px 27px),
                    radial-gradient(circle at 35% 35%, #388c8b, #103c54 72%);
                box-shadow: inset -20px -12px 38px rgba(4, 27, 42, .55), 0 0 32px rgba(27, 153, 150, .2);
                transform: translate(-50%, -50%);
                animation: turn 18s linear infinite;
            }
            .fallback-globe::after {
                position: absolute;
                inset: 20% -12%;
                border: 1px solid rgba(22, 141, 146, .7);
                border-radius: 50%;
                content: '';
                transform: rotate(-24deg);
            }
            .fallback-globe.hidden { display: none; }
            @keyframes turn { to { transform: translate(-50%, -50%) rotate(360deg); } }
            @media (max-width: 600px) {
                .hero { height: 248px; }
                .copy { left: 20px; max-width: 58%; }
                h2 { font-size: 24px; }
                .description { max-width: 235px; font-size: 12px; }
                .coordinates { font-size: 9px; }
                #scene { left: 40%; }
                .fallback-globe { left: 65%; width: 150px; }
            }
            @media (prefers-reduced-motion: reduce) {
                .fallback-globe { animation: none; }
            }
        </style>
        </head>
        <body>
            <section class="hero">
                <div class="copy">
                    <div class="eyebrow"><span class="pulse"></span> GEOSPATIAL INTELLIGENCE</div>
                    <h2>See your land<br>in a new dimension.</h2>
                    <p class="description">AI-assisted parcel mapping, built for a clearer view of every boundary.</p>
                    <div class="coordinates">16.50° N &nbsp; 78.10° E &nbsp; / &nbsp; LIVE MODEL</div>
                </div>
                <div id="scene" aria-label="Animated 3D globe showing connected parcel locations"></div>
                <div class="fallback-globe" id="fallback"></div>
            </section>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script>
                (function () {
                    if (!window.THREE) return;

                    const sceneElement = document.getElementById('scene');
                    const fallback = document.getElementById('fallback');
                    const scene = new THREE.Scene();
                    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
                    camera.position.z = 5.2;

                    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
                    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
                    renderer.setClearColor(0x000000, 0);
                    sceneElement.appendChild(renderer.domElement);
                    fallback.classList.add('hidden');

                    const globe = new THREE.Group();
                    scene.add(globe);
                    const textureCanvas = document.createElement('canvas');
                    textureCanvas.width = 1024;
                    textureCanvas.height = 512;
                    const context = textureCanvas.getContext('2d');
                    context.fillStyle = '#103d55';
                    context.fillRect(0, 0, 1024, 512);
                    context.strokeStyle = 'rgba(105, 206, 190, .22)';
                    context.lineWidth = 1;
                    for (let x = 0; x <= 1024; x += 64) {
                        context.beginPath(); context.moveTo(x, 0); context.lineTo(x, 512); context.stroke();
                    }
                    for (let y = 0; y <= 512; y += 64) {
                        context.beginPath(); context.moveTo(0, y); context.lineTo(1024, y); context.stroke();
                    }
                    const landShapes = [
                        [[80,105],[118,65],[180,52],[214,83],[198,117],[220,143],[192,166],[182,205],[149,216],[130,178],[105,164],[90,132]],
                        [[222,245],[258,227],[286,246],[298,282],[282,325],[269,372],[243,417],[224,380],[228,336],[210,298]],
                        [[430,112],[466,94],[501,108],[511,142],[493,169],[464,158],[444,140]],
                        [[455,178],[505,166],[535,196],[526,246],[505,295],[480,322],[461,281],[443,228]],
                        [[506,104],[553,81],[625,85],[681,103],[735,101],[797,129],[843,163],[827,195],[786,190],[762,220],[723,211],[695,238],[668,213],[637,220],[611,193],[570,190],[547,155]],
                        [[788,274],[824,259],[856,272],[868,302],[844,327],[807,317]],
                        [[330,64],[372,45],[403,60],[397,90],[363,101],[339,87]]
                    ];
                    landShapes.forEach(function (shape) {
                        context.beginPath();
                        context.moveTo(shape[0][0], shape[0][1]);
                        shape.slice(1).forEach(function (point) { context.lineTo(point[0], point[1]); });
                        context.closePath();
                        context.fillStyle = '#32877e';
                        context.fill();
                        context.strokeStyle = '#80c4a7';
                        context.lineWidth = 2;
                        context.stroke();
                    });

                    const globeTexture = new THREE.CanvasTexture(textureCanvas);
                    globeTexture.colorSpace = THREE.SRGBColorSpace;
                    const earth = new THREE.Mesh(
                        new THREE.SphereGeometry(1.45, 72, 48),
                        new THREE.MeshPhongMaterial({ map: globeTexture, shininess: 18, specular: 0x438c83 })
                    );
                    globe.add(earth);
                    globe.add(new THREE.Mesh(
                        new THREE.SphereGeometry(1.47, 48, 32),
                        new THREE.MeshBasicMaterial({ color: 0x91e3cf, wireframe: true, transparent: true, opacity: 0.12 })
                    ));

                    scene.add(new THREE.AmbientLight(0xa7e7d8, 1.7));
                    const keyLight = new THREE.PointLight(0xffffff, 2.4);
                    keyLight.position.set(-3, 3, 5);
                    scene.add(keyLight);

                    const orbitMaterial = new THREE.MeshBasicMaterial({ color: 0x168f99, transparent: true, opacity: 0.55 });
                    const orbit = new THREE.Mesh(new THREE.TorusGeometry(1.78, 0.008, 8, 120), orbitMaterial);
                    orbit.rotation.set(1.18, 0.12, -0.28);
                    globe.add(orbit);

                    function surfacePoint(latitude, longitude, radius) {
                        const phi = (90 - latitude) * Math.PI / 180;
                        const theta = (longitude + 180) * Math.PI / 180;
                        return new THREE.Vector3(
                            -radius * Math.sin(phi) * Math.cos(theta),
                            radius * Math.cos(phi),
                            radius * Math.sin(phi) * Math.sin(theta)
                        );
                    }

                    const locations = [
                        [16.5, 78.1], [51.5, -0.1], [37.8, -122.4], [-23.5, 133.8]
                    ];
                    const markers = [];
                    locations.forEach(function (location, index) {
                        const marker = new THREE.Mesh(
                            new THREE.SphereGeometry(0.035, 16, 12),
                            new THREE.MeshBasicMaterial({ color: index === 0 ? 0xffb45d : 0x87f0d3 })
                        );
                        marker.position.copy(surfacePoint(location[0], location[1], 1.49));
                        globe.add(marker);
                        markers.push(marker);
                    });

                    for (let index = 0; index < locations.length - 1; index++) {
                        const start = surfacePoint(locations[index][0], locations[index][1], 1.5);
                        const end = surfacePoint(locations[index + 1][0], locations[index + 1][1], 1.5);
                        const middle = start.clone().add(end).normalize().multiplyScalar(1.9);
                        const curve = new THREE.QuadraticBezierCurve3(start, middle, end);
                        globe.add(new THREE.Line(
                            new THREE.BufferGeometry().setFromPoints(curve.getPoints(48)),
                            new THREE.LineBasicMaterial({ color: 0xffb45d, transparent: true, opacity: 0.75 })
                        ));
                    }

                    const starGeometry = new THREE.BufferGeometry();
                    const starPositions = [];
                    for (let index = 0; index < 180; index++) {
                        starPositions.push((Math.random() - 0.5) * 6, (Math.random() - 0.5) * 3.5, -1 - Math.random() * 2);
                    }
                    starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starPositions, 3));
                    scene.add(new THREE.Points(starGeometry, new THREE.PointsMaterial({ color: 0x398f91, size: 0.012, transparent: true, opacity: 0.55 })));

                    function resize() {
                        const width = sceneElement.clientWidth;
                        const height = sceneElement.clientHeight;
                        if (!width || !height) return;
                        renderer.setSize(width, height, false);
                        camera.aspect = width / height;
                        camera.updateProjectionMatrix();
                    }
                    window.addEventListener('resize', resize);
                    resize();

                    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
                    function animate(time) {
                        requestAnimationFrame(animate);
                        if (!reducedMotion) {
                            globe.rotation.y += 0.0018;
                            globe.rotation.x = Math.sin(time * 0.00025) * 0.045;
                            markers.forEach(function (marker, index) {
                                const pulse = 1 + Math.sin(time * 0.004 + index) * 0.18;
                                marker.scale.setScalar(pulse);
                            });
                        }
                        renderer.render(scene, camera);
                    }
                    requestAnimationFrame(animate);
                })();
            </script>
        </body>
        </html>
        """,
        height=292,
        scrolling=False
    )


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "project": None,
    "parcels": [],
    "buildings": [],
    "roads": [],
    "history": [],
    "analysis_done": False,
    "selected_parcel": None,
    "uploaded_file": None,
    "cadastral_gdf": None,
    "cadastral_source": None,
    "cadastral_stats": {},
    "drone_raster": None,
    "drone_raster_meta": {},
    "drone_boundary_coordinates": None,
    "comparison_results": None,
    "editing": False,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# DEMO DATA
# ============================================================

def create_demo_data():

    parcels = [

        {
            "parcel_id": "GV-P-0001",
            "confidence": 94,
            "status": "Candidate",
            "qa": ["Valid geometry"],
            "area": 428.52,
            "perimeter": 84.21,
            "coordinates": [
                [78.100, 16.500],
                [78.103, 16.500],
                [78.103, 16.503],
                [78.1005, 16.5035],
                [78.100, 16.502],
                [78.100, 16.500]
            ]
        },

        {
            "parcel_id": "GV-P-0002",
            "confidence": 72,
            "status": "Candidate",
            "qa": ["Possible overlap"],
            "area": 512.31,
            "perimeter": 91.24,
            "coordinates": [
                [78.103, 16.500],
                [78.106, 16.5005],
                [78.106, 16.503],
                [78.103, 16.503],
                [78.103, 16.500]
            ]
        },

        {
            "parcel_id": "GV-P-0003",
            "confidence": 57,
            "status": "Candidate",
            "qa": [
                "Low confidence",
                "Irregular geometry"
            ],
            "area": 398.72,
            "perimeter": 87.52,
            "coordinates": [
                [78.1005, 16.5035],
                [78.103, 16.503],
                [78.103, 16.506],
                [78.101, 16.5065],
                [78.1005, 16.5035]
            ]
        },

        {
            "parcel_id": "GV-P-0004",
            "confidence": 88,
            "status": "Candidate",
            "qa": ["Edge artifact"],
            "area": 463.42,
            "perimeter": 88.31,
            "coordinates": [
                [78.103, 16.503],
                [78.106, 16.503],
                [78.1055, 16.506],
                [78.103, 16.506],
                [78.103, 16.503]
            ]
        },

        {
            "parcel_id": "GV-P-0005",
            "confidence": 43,
            "status": "Candidate",
            "qa": [
                "Low confidence",
                "Possible overlap"
            ],
            "area": 331.10,
            "perimeter": 80.42,
            "coordinates": [
                [78.101, 16.5065],
                [78.103, 16.506],
                [78.103, 16.509],
                [78.1005, 16.509],
                [78.101, 16.5065]
            ]
        },

        {
            "parcel_id": "GV-P-0006",
            "confidence": 67,
            "status": "Candidate",
            "qa": ["Possible overlap"],
            "area": 489.62,
            "perimeter": 92.11,
            "coordinates": [
                [78.103, 16.506],
                [78.1055, 16.506],
                [78.106, 16.509],
                [78.103, 16.509],
                [78.103, 16.506]
            ]
        }
    ]

    for parcel in parcels:

        if parcel["confidence"] >= 85:

            parcel["bucket"] = "HIGH CONFIDENCE"

        elif parcel["confidence"] >= 60:

            parcel["bucket"] = "INSPECT"

        else:

            parcel["bucket"] = "FIELD CHECK"

    buildings = [

        {
            "building_id": "B-001",
            "coordinates": [
                [78.1005, 16.501],
                [78.1015, 16.501],
                [78.1015, 16.502],
                [78.1005, 16.502],
                [78.1005, 16.501]
            ]
        },

        {
            "building_id": "B-002",
            "coordinates": [
                [78.1037, 16.501],
                [78.1048, 16.501],
                [78.1048, 16.502],
                [78.1037, 16.502],
                [78.1037, 16.501]
            ]
        },

        {
            "building_id": "B-003",
            "coordinates": [
                [78.1017, 16.504],
                [78.1025, 16.504],
                [78.1025, 16.505],
                [78.1017, 16.505],
                [78.1017, 16.504]
            ]
        },

        {
            "building_id": "B-004",
            "coordinates": [
                [78.104, 16.504],
                [78.105, 16.504],
                [78.105, 16.505],
                [78.104, 16.505],
                [78.104, 16.504]
            ]
        }
    ]

    roads = [

        {
            "road_id": "R-001",
            "coordinates": [
                [78.098, 16.504],
                [78.108, 16.504]
            ]
        },

        {
            "road_id": "R-002",
            "coordinates": [
                [78.103, 16.498],
                [78.103, 16.511]
            ]
        },

        {
            "road_id": "R-003",
            "coordinates": [
                [78.098, 16.508],
                [78.108, 16.508]
            ]
        }
    ]

    return parcels, buildings, roads


# ============================================================
# LOAD DEMO
# ============================================================

def load_demo():

    parcels, buildings, roads = create_demo_data()

    st.session_state.project = {

        "id": "GV-DEMO-001",

        "name":
            "Demo Urban Parcel Project",

        "location":
            "Synthetic Urban Area, India",

        "crs":
            "EPSG:4326",

        "created":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M"
            )
    }

    st.session_state.parcels = parcels

    st.session_state.buildings = buildings

    st.session_state.roads = roads

    st.session_state.analysis_done = False

    st.session_state.history = []

    st.session_state.history.append({

        "time":
            datetime.now().strftime("%H:%M:%S"),

        "parcel":
            "SYSTEM",

        "action":
            "Demo project loaded",

        "user":
            "System"
    })


# ============================================================
# GENERAL HELPERS
# ============================================================

def find_parcel(parcel_id):

    for parcel in st.session_state.parcels:

        if parcel["parcel_id"] == parcel_id:

            return parcel

    return None


def add_history(parcel, action):

    st.session_state.history.append({

        "time":
            datetime.now().strftime("%H:%M:%S"),

        "parcel":
            parcel,

        "action":
            action,

        "user":
            "Demo Surveyor"
    })


def confidence_bucket(confidence):

    if confidence >= 85:

        return "HIGH CONFIDENCE"

    if confidence >= 60:

        return "INSPECT"

    return "FIELD CHECK"


def confidence_badge(bucket):

    if bucket == "HIGH CONFIDENCE":

        return (
            '<span class="high">'
            'HIGH CONFIDENCE'
            '</span>'
        )

    if bucket == "INSPECT":

        return (
            '<span class="inspect">'
            'INSPECT'
            '</span>'
        )

    return (
        '<span class="field">'
        'FIELD CHECK'
        '</span>'
    )


# ============================================================
# UTM CRS
# ============================================================

def get_utm_crs(latitude, longitude):

    zone = int(
        (longitude + 180) / 6
    ) + 1

    if latitude >= 0:

        epsg = 32600 + zone

    else:

        epsg = 32700 + zone

    return f"EPSG:{epsg}"


# ============================================================
# DRONE GEOTIFF
# ============================================================

def normalize_raster_for_display(array):

    if array.ndim == 2:

        array = array[
            np.newaxis,
            ...
        ]

    bands = array.shape[0]

    if bands >= 3:

        rgb = array[:3].astype(
            "float32"
        )

    else:

        rgb = np.repeat(
            array[:1].astype("float32"),
            3,
            axis=0
        )

    output = np.zeros_like(
        rgb,
        dtype="float32"
    )

    for i in range(3):

        band = rgb[i]

        finite = band[
            np.isfinite(band)
        ]

        if finite.size == 0:

            continue

        low, high = np.percentile(
            finite,
            [2, 98]
        )

        if high <= low:

            low = float(
                finite.min()
            )

            high = float(
                finite.max()
            )

        if high > low:

            output[i] = (
                band - low
            ) / (
                high - low
            )

    output = (
        np.clip(
            output,
            0,
            1
        ) * 255
    ).astype(
        "uint8"
    )

    return np.moveaxis(
        output,
        0,
        -1
    )


def create_edge_suggestion_overlay(
    image,
    sensitivity=97
):

    overlay = np.zeros(
        (*image.shape[:2], 4),
        dtype="uint8"
    )

    if min(image.shape[:2]) < 2:

        return overlay

    grayscale = np.mean(
        image.astype("float32"),
        axis=2
    )

    gradient_y, gradient_x = np.gradient(
        grayscale
    )

    gradient_strength = np.hypot(
        gradient_x,
        gradient_y
    )

    valid_gradients = gradient_strength[
        np.isfinite(gradient_strength)
    ]

    if valid_gradients.size == 0 or valid_gradients.max() < 1:

        return overlay

    threshold = np.percentile(
        valid_gradients,
        sensitivity
    )

    edges = (
        (gradient_strength >= threshold)
        &
        (gradient_strength >= 1)
    )

    overlay[edges] = [
        255,
        74,
        48,
        220
    ]

    return overlay


def load_drone_geotiff(file_bytes):

    if not RASTERIO_AVAILABLE:

        raise RuntimeError(
            "rasterio is not installed."
        )

    with MemoryFile(
        file_bytes
    ) as memory_file:

        with memory_file.open() as src:

            data = src.read()

            image = normalize_raster_for_display(
                data
            )

            map_image = None
            map_bounds = None

            if src.crs:

                from rasterio.transform import array_bounds
                from rasterio.warp import (
                    Resampling,
                    calculate_default_transform,
                    reproject
                )

                scale = max(
                    1,
                    src.width / 1200,
                    src.height / 1200
                )

                map_width = max(
                    1,
                    int(src.width / scale)
                )

                map_height = max(
                    1,
                    int(src.height / scale)
                )

                map_transform, map_width, map_height = (
                    calculate_default_transform(
                        src.crs,
                        "EPSG:4326",
                        src.width,
                        src.height,
                        *src.bounds,
                        dst_width=map_width,
                        dst_height=map_height
                    )
                )

                map_image = np.zeros(
                    (map_height, map_width, 3),
                    dtype="uint8"
                )

                for band in range(3):

                    reproject(
                        source=image[:, :, band],
                        destination=map_image[:, :, band],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=map_transform,
                        dst_crs="EPSG:4326",
                        resampling=Resampling.bilinear
                    )

                map_bounds = array_bounds(
                    map_height,
                    map_width,
                    map_transform
                )

            metadata = {

                "width":
                    src.width,

                "height":
                    src.height,

                "bands":
                    src.count,

                "dtype":
                    str(
                        src.dtypes[0]
                    ),

                "crs":
                    str(src.crs)
                    if src.crs
                    else "Missing CRS",

                "bounds":
                    tuple(src.bounds),

                "resolution":
                    src.res,

                "nodata":
                    src.nodata,

                "driver":
                    src.driver,

                "map_image":
                    map_image,

                "map_bounds":
                    map_bounds
            }

    return image, metadata


# ============================================================
# DRONE LAND SURVEY FUNCTIONS
# ============================================================

def calculate_polygon_measurements(
    polygon,
    projected_crs
):

    gdf = gpd.GeoDataFrame(
        geometry=[polygon],
        crs="EPSG:4326"
    )

    projected = gdf.to_crs(
        projected_crs
    )

    geometry = projected.geometry.iloc[0]

    area_m2 = geometry.area

    perimeter_m = geometry.length

    hectares = (
        area_m2 / 10000
    )

    acres = (
        area_m2 *
        0.000247105381
    )

    return (
        area_m2,
        hectares,
        acres,
        perimeter_m
    )


def calculate_side_lengths(
    polygon,
    projected_crs
):

    gdf = gpd.GeoDataFrame(
        geometry=[polygon],
        crs="EPSG:4326"
    )

    projected = gdf.to_crs(
        projected_crs
    )

    coordinates = list(
        projected.geometry.iloc[
            0
        ].exterior.coords
    )

    sides = []

    for i in range(
        len(coordinates) - 1
    ):

        x1, y1 = coordinates[i]

        x2, y2 = coordinates[i + 1]

        distance = math.sqrt(

            (x2 - x1) ** 2
            +
            (y2 - y1) ** 2
        )

        sides.append(
            distance
        )

    return sides


# ============================================================
# CADASTRAL FUNCTIONS
# ============================================================

def extract_cadastral_zip(
    zip_bytes
):

    temp_dir = tempfile.mkdtemp(
        prefix="geovision_cadastral_"
    )

    zip_path = os.path.join(
        temp_dir,
        "cadastral.zip"
    )

    with open(
        zip_path,
        "wb"
    ) as file:

        file.write(
            zip_bytes
        )

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as archive:

        archive.extractall(
            temp_dir
        )

    candidates = []

    for walk_result in os.walk(
        temp_dir
    ):

        root = walk_result[0]
        files = walk_result[2]

        for name in files:

            if name.lower().endswith(
                (
                    ".shp",
                    ".geojson",
                    ".json",
                    ".kml",
                    ".gpkg"
                )
            ):

                candidates.append(
                    os.path.join(
                        root,
                        name
                    )
                )

    candidates.sort(
        key=lambda path:
        0
        if path.lower().endswith(".shp")
        else 1
    )

    if not candidates:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        raise ValueError(
            "ZIP does not contain a supported GIS file."
        )

    return (
        temp_dir,
        candidates[0]
    )


def load_cadastral_dataset(
    filename,
    file_bytes
):

    lower = filename.lower()

    if lower.endswith(
        ".zip"
    ):

        temp_dir, path = (
            extract_cadastral_zip(
                file_bytes
            )
        )

    else:

        temp_dir = tempfile.mkdtemp(
            prefix="geovision_cadastral_"
        )

        path = os.path.join(
            temp_dir,
            os.path.basename(
                filename
            )
        )

        with open(
            path,
            "wb"
        ) as file:

            file.write(
                file_bytes
            )

    if path.lower().endswith(
        ".kml"
    ):

        gdf = gpd.read_file(
            path,
            driver="KML"
        )

    else:

        gdf = gpd.read_file(
            path
        )

    return (
        gdf,
        os.path.basename(path),
        temp_dir
    )


def process_cadastral_gdf(
    gdf
):

    if gdf is None or gdf.empty:

        raise ValueError(
            "Cadastral dataset contains no features."
        )

    gdf = gdf.copy()

    gdf = gdf[
        gdf.geometry.notna()
    ]

    gdf = gdf[
        ~gdf.geometry.is_empty
    ]

    if gdf.empty:

        raise ValueError(
            "No valid geometries found."
        )

    invalid_count = int(
        (~gdf.geometry.is_valid).sum()
    )

    if invalid_count > 0:

        try:

            gdf.geometry = (
                gdf.geometry.make_valid()
            )

        except Exception:

            gdf.geometry = (
                gdf.geometry.buffer(0)
            )

    if gdf.crs is None:

        st.warning(
            "Cadastral CRS missing. "
            "Assuming EPSG:4326."
        )

        gdf = gdf.set_crs(
            epsg=4326
        )

    wgs84 = gdf.to_crs(
        epsg=4326
    )

    centroid = (
        wgs84.geometry
        .unary_union
        .centroid
    )

    longitude = centroid.x

    latitude = centroid.y

    if (
        -180 <= longitude <= 180
        and
        -80 <= latitude <= 84
    ):

        zone = int(
            (longitude + 180) // 6
        ) + 1

        if latitude >= 0:

            epsg = 32600 + zone

        else:

            epsg = 32700 + zone

    else:

        epsg = 3857

    projected = wgs84.to_crs(
        epsg=epsg
    )

    wgs84["area_m2"] = (
        projected.geometry.area
    )

    wgs84["area_hectares"] = (
        wgs84["area_m2"] / 10000
    )

    wgs84["area_acres"] = (
        wgs84["area_m2"]
        * 0.000247105381
    )

    wgs84["perimeter_m"] = (
        projected.geometry.length
    )

    stats = {

        "total":
            len(wgs84),

        "invalid_fixed":
            invalid_count,

        "source_crs":
            str(gdf.crs),

        "output_crs":
            "EPSG:4326",

        "area_crs":
            f"EPSG:{epsg}",

        "min_area":
            float(
                wgs84["area_m2"].min()
            ),

        "max_area":
            float(
                wgs84["area_m2"].max()
            ),

        "avg_area":
            float(
                wgs84["area_m2"].mean()
            )
    }

    return (
        wgs84,
        stats
    )


# ============================================================
# AI PARCEL → GEODATAFRAME
# ============================================================

def ai_parcels_to_gdf():

    records = []

    for parcel in st.session_state.parcels:

        coordinates = (
            parcel.get(
                "coordinates"
            )
        )

        if not coordinates:

            continue

        try:

            polygon = Polygon(
                coordinates
            )

            if not polygon.is_valid:

                polygon = polygon.buffer(
                    0
                )

            if polygon.is_empty:

                continue

            records.append({

                "parcel_id":
                    parcel["parcel_id"],

                "confidence":
                    parcel["confidence"],

                "status":
                    parcel["status"],

                "geometry":
                    polygon
            })

        except Exception:

            continue

    if not records:

        return gpd.GeoDataFrame(
            columns=[
                "parcel_id",
                "confidence",
                "status",
                "geometry"
            ],
            geometry="geometry",
            crs="EPSG:4326"
        )

    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:4326"
    )


# ============================================================
# SPATIAL COMPARISON
# ============================================================

def compare_ai_with_cadastral():

    cadastral = (
        st.session_state.cadastral_gdf
    )

    ai = ai_parcels_to_gdf()

    if cadastral is None:

        raise ValueError(
            "Import cadastral data first."
        )

    if ai.empty:

        raise ValueError(
            "No AI parcels available."
        )

    cadastral = cadastral.copy()

    ai = ai.copy()

    if cadastral.crs is None:

        cadastral = cadastral.set_crs(
            epsg=4326
        )

    cadastral = cadastral.to_crs(
        epsg=4326
    )

    combined = pd.concat(
        [
            cadastral[
                ["geometry"]
            ],

            ai[
                ["geometry"]
            ]
        ],
        ignore_index=True
    )

    combined = gpd.GeoDataFrame(
        combined,
        geometry="geometry",
        crs="EPSG:4326"
    )

    metric_crs = (
        combined.estimate_utm_crs()
        or "EPSG:3857"
    )

    cadastral_m = (
        cadastral.to_crs(
            metric_crs
        )
    )

    ai_m = (
        ai.to_crs(
            metric_crs
        )
    )

    cadastral_areas = (
        cadastral_m.geometry.area
    )

    results = []

    for row_data in (
        ai_m.iterrows()
    ):

        ai_row = row_data[1]

        best_iou = 0

        best_overlap = 0

        best_index = None

        ai_area = (
            ai_row.geometry.area
        )

        for cad_index, cad_row in (
            cadastral_m.iterrows()
        ):

            try:

                intersection = (
                    ai_row.geometry
                    .intersection(
                        cad_row.geometry
                    )
                    .area
                )

                if intersection <= 0:

                    continue

                cad_area = float(
                    cadastral_areas.loc[
                        cad_index
                    ]
                )

                union = (
                    ai_area
                    +
                    cad_area
                    -
                    intersection
                )

                iou = (
                    intersection / union
                    if union > 0
                    else 0
                )

                if iou > best_iou:

                    best_iou = iou

                    best_overlap = (
                        intersection /
                        ai_area *
                        100
                        if ai_area > 0
                        else 0
                    )

                    best_index = (
                        cad_index
                    )

            except Exception:

                continue

        cadastral_id = "No match"

        if best_index is not None:

            non_geometry = [

                column
                for column in cadastral.columns
                if column != "geometry"

            ]

            if non_geometry:

                cadastral_id = str(
                    cadastral.iloc[
                        best_index
                    ][
                        non_geometry[0]
                    ]
                )

            else:

                cadastral_id = (
                    f"CAD-{best_index + 1}"
                )

        if best_iou >= 0.70:

            status = "Strong overlap"

        elif best_iou >= 0.30:

            status = "Partial overlap"

        else:

            status = "Mismatch / review"

        results.append({

            "AI Parcel":
                ai_row["parcel_id"],

            "AI Confidence (%)":
                ai_row["confidence"],

            "Cadastral Parcel":
                cadastral_id,

            "IoU (%)":
                round(
                    best_iou * 100,
                    2
                ),

            "AI Overlap (%)":
                round(
                    best_overlap,
                    2
                ),

            "Status":
                status
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(
    output_path,
    survey_data
):

    if not REPORTLAB_AVAILABLE:

        raise RuntimeError(
            "reportlab is not installed."
        )

    document = SimpleDocTemplate(
        output_path,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "<b>GEOVISION AI SURVEY REPORT</b>",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    table_data = [

        ["Parameter", "Value"],

        [
            "Survey Type",
            "Drone / Aerial Survey"
        ],

        [
            "CRS",
            survey_data["crs"]
        ],

        [
            "Area (m²)",
            f'{survey_data["area_m2"]:.2f}'
        ],

        [
            "Hectares",
            f'{survey_data["hectares"]:.4f}'
        ],

        [
            "Acres",
            f'{survey_data["acres"]:.4f}'
        ],

        [
            "Perimeter (m)",
            f'{survey_data["perimeter"]:.2f}'
        ],

        [
            "Boundary Points",
            survey_data["points"]
        ]
    ]

    table = Table(
        table_data,
        colWidths=[
            220,
            250
        ]
    )

    table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                1,
                colors.black
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            )
        ])
    )

    story.append(
        table
    )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Accuracy depends on drone imagery quality, "
            "GNSS/RTK/PPK accuracy, GCPs, orthomosaic "
            "quality, CRS and boundary marking.",
            styles["BodyText"]
        )
    )

    document.build(
        story
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🌍 GeoVision AI"
)

st.sidebar.caption(
    "AI-Powered Urban Parcel Mapping"
)

st.sidebar.divider()

pages = [

    "Dashboard",

    "Projects",

    "Imagery",

    "Drone GeoTIFF",

    "Cadastral Import",

    "Spatial Comparison",

    "AI Analysis",

    "Parcel Review",

    "Field Check",

    "Exports",

    "Architecture",

    "Settings"
]

page = st.sidebar.radio(
    "Navigation",
    pages
)

st.sidebar.divider()

if st.sidebar.button(
    "🚀 Load Demo Project",
    use_container_width=True
):

    load_demo()

    st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    if not st.session_state.project:

        load_demo()

    st.markdown(
        '<div class="main-title">'
        'GeoVision AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Based Urban Parcel Mapping, Cadastral '
        'Feature Extraction & Drone Land Survey'
        '</div>',
        unsafe_allow_html=True
    )

    render_animated_globe()

    st.divider()

    parcels = (
        st.session_state.parcels
    )

    total = len(parcels)

    high = len([
        p for p in parcels
        if p["confidence"] >= 85
    ])

    inspect = len([
        p for p in parcels
        if 60 <= p["confidence"] < 85
    ])

    field = len([
        p for p in parcels
        if p["confidence"] < 60
    ])

    accepted = len([
        p for p in parcels
        if p["status"] == "Accepted"
    ])

    rejected = len([
        p for p in parcels
        if p["status"] == "Rejected"
    ])

    edited = len([
        p for p in parcels
        if p["status"] == "Edited"
    ])

    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )

    c1.metric(
        "Projects",
        1
    )

    c2.metric(
        "Candidate Parcels",
        total
    )

    c3.metric(
        "High Confidence",
        high
    )

    c4.metric(
        "Needs Inspection",
        inspect
    )

    c5.metric(
        "Field Check",
        field
    )

    st.divider()

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Accepted",
        accepted
    )

    c2.metric(
        "Edited",
        edited
    )

    c3.metric(
        "Rejected",
        rejected
    )

    c4.metric(
        "Buildings Detected",
        len(
            st.session_state.buildings
        )
    )

    st.divider()

    st.info(
        "Candidate parcel boundaries are AI-assisted "
        "outputs and should be verified against "
        "authoritative cadastral records."
    )


# ============================================================
# PROJECTS
# ============================================================

elif page == "Projects":

    st.title(
        "📁 Projects"
    )

    with st.form(
        "project_form"
    ):

        name = st.text_input(
            "Project Name",
            "Urban Parcel Project"
        )

        location = st.text_input(
            "Location",
            "Narasaraopet, Andhra Pradesh"
        )

        description = st.text_area(
            "Description",
            "AI-assisted urban parcel mapping"
        )

        crs = st.text_input(
            "CRS",
            "EPSG:4326"
        )

        submit = st.form_submit_button(
            "Create Project"
        )

        if submit:

            st.session_state.project = {

                "id":
                    "GV-" +
                    str(
                        random.randint(
                            1000,
                            9999
                        )
                    ),

                "name":
                    name,

                "location":
                    location,

                "description":
                    description,

                "crs":
                    crs,

                "created":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    )
            }

            st.success(
                "Project created."
            )

    if st.session_state.project:

        project = st.session_state.project

        st.subheader(
            project.get("name", "Current Project")
        )

        st.caption(
            f"Project ID: {project.get('id', 'N/A')}"
        )

        project_col1, project_col2, project_col3 = st.columns(3)

        project_col1.markdown("**Location**")
        project_col1.write(project.get("location") or "Not specified")

        project_col2.markdown("**Coordinate Reference System**")
        project_col2.write(project.get("crs") or "Not specified")

        project_col3.markdown("**Created**")
        project_col3.write(project.get("created") or "Not specified")

        if project.get("description"):
            st.markdown("**Description**")
            st.write(project["description"])


# ============================================================
# IMAGERY
# ============================================================

elif page == "Imagery":

    st.title(
        "📷 Imagery"
    )

    uploaded = st.file_uploader(
        "Upload imagery",
        type=[
            "tif",
            "tiff",
            "png",
            "jpg",
            "jpeg",
            "geojson"
        ]
    )

    if uploaded:

        st.session_state.uploaded_file = (
            uploaded
        )

        st.success(
            f"{uploaded.name} uploaded."
        )

        st.write(
            "Size:",
            round(
                uploaded.size / 1024,
                2
            ),
            "KB"
        )

        if uploaded.name.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg"
            )
        ):

            st.image(
                uploaded,
                use_container_width=True
            )

        else:

            st.info(
                "Use Drone GeoTIFF for "
                "georeferenced survey processing."
            )


# ============================================================
# DRONE GEOTIFF
# ============================================================

elif page == "Drone GeoTIFF":

    st.title(
        "🚁 Drone GeoTIFF Land Survey"
    )

    st.info(
        """
        Upload a georeferenced drone/aerial GeoTIFF.
        The application automatically reads the CRS and
        selects an appropriate UTM CRS for metric measurements.
        """
    )

    if not RASTERIO_AVAILABLE:

        st.error(
            "Install rasterio first:"
        )

        st.code(
            "pip install rasterio"
        )

        st.stop()

    uploaded = st.file_uploader(
        "Upload Drone Orthomosaic GeoTIFF",
        type=[
            "tif",
            "tiff"
        ],
        key="drone_upload"
    )

    if uploaded:

        if st.button(
            "🚀 Load GeoTIFF",
            use_container_width=True
        ):

            try:

                image, metadata = (
                    load_drone_geotiff(
                        uploaded.getvalue()
                    )
                )

                st.session_state.drone_raster = (
                    image
                )

                st.session_state.drone_raster_meta = (
                    metadata
                )

                st.success(
                    "GeoTIFF loaded successfully."
                )

            except Exception as e:

                st.error(
                    f"GeoTIFF error: {e}"
                )

    image = (
        st.session_state.drone_raster
    )

    metadata = (
        st.session_state.drone_raster_meta
    )

    if image is None:

        st.warning(
            "Upload a GeoTIFF to begin."
        )

        st.stop()

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Width",
        metadata["width"]
    )

    c2.metric(
        "Height",
        metadata["height"]
    )

    c3.metric(
        "Bands",
        metadata["bands"]
    )

    c4.metric(
        "Format",
        metadata["driver"]
    )

    st.write(
        "**CRS:**",
        metadata["crs"]
    )

    st.write(
        "**Bounds:**",
        metadata["bounds"]
    )

    st.write(
        "**Resolution:**",
        metadata["resolution"]
    )

    if metadata["crs"] == "Missing CRS":

        st.error(
            "This GeoTIFF has no CRS. "
            "Real-world area calculation should "
            "not be performed until it is correctly "
            "georeferenced."
        )

        st.image(
            image,
            use_container_width=True
        )

        st.stop()

    st.subheader(
        "📐 Land Boundary Measurement"
    )

    st.info(
        """
        Draw the parcel boundary on the georeferenced image. If cadastral data
        has been imported, its boundaries are overlaid for reference. Optional
        image-edge suggestions show contrast changes, not predicted parcel or
        legal boundaries.
        """
    )

    if not FOLIUM_AVAILABLE:

        st.error(
            "Interactive boundary drawing requires folium and streamlit-folium. "
            "Install the project requirements and reload the app."
        )

        st.stop()

    map_bounds = metadata.get("map_bounds")
    map_image = metadata.get("map_image")

    if map_bounds is None or map_image is None:

        st.error(
            "The GeoTIFF could not be placed on a WGS84 map. "
            "Check that its CRS and georeferencing are valid."
        )

        st.stop()

    west, south, east, north = map_bounds

    parcel_map = folium.Map(
        location=[
            (south + north) / 2,
            (west + east) / 2
        ],
        zoom_start=18,
        tiles=None,
        control_scale=True
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri and its contributors",
        name="Satellite imagery",
        max_zoom=22
    ).add_to(parcel_map)

    folium.raster_layers.ImageOverlay(
        image=map_image,
        bounds=[
            [south, west],
            [north, east]
        ],
        opacity=0.9,
        name="Uploaded orthomosaic",
        interactive=True
    ).add_to(parcel_map)

    show_edge_suggestions = st.checkbox(
        "Show experimental image-edge suggestions",
        value=True
    )

    if show_edge_suggestions:

        edge_sensitivity = st.slider(
            "Edge sensitivity (higher shows fewer edges)",
            min_value=90,
            max_value=99,
            value=97
        )

        edge_overlay = create_edge_suggestion_overlay(
            map_image,
            edge_sensitivity
        )

        folium.raster_layers.ImageOverlay(
            image=edge_overlay,
            bounds=[
                [south, west],
                [north, east]
            ],
            opacity=0.9,
            name="Experimental image-edge suggestions",
            interactive=False
        ).add_to(parcel_map)

    cadastral_gdf = st.session_state.cadastral_gdf

    if cadastral_gdf is not None:

        cadastral_map_gdf = cadastral_gdf.to_crs(
            "EPSG:4326"
        ).head(1000)

        folium.GeoJson(
            cadastral_map_gdf.__geo_interface__,
            name="Cadastral parcels",
            style_function=lambda feature: {
                "color": "#ffb000",
                "weight": 2,
                "fillOpacity": 0.05
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["area_m2"],
                aliases=["Cadastral area (m²):"]
            ) if "area_m2" in cadastral_map_gdf.columns else None
        ).add_to(parcel_map)

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "polygon": {
                "allowIntersection": False,
                "showArea": True,
                "shapeOptions": {
                    "color": "#ff5533"
                }
            },
            "rectangle": False,
            "circle": False,
            "circlemarker": False,
            "marker": False
        },
        edit_options={
            "edit": True,
            "remove": True
        }
    ).add_to(parcel_map)

    folium.LayerControl().add_to(parcel_map)

    map_state = st_folium(
        parcel_map,
        height=600,
        use_container_width=True,
        returned_objects=["all_drawings"],
        key="drone_boundary_map"
    )

    if map_state and map_state.get("all_drawings") is not None:

        polygon_features = [
            feature
            for feature in map_state["all_drawings"]
            if feature.get("geometry", {}).get("type") == "Polygon"
        ]

        st.session_state.drone_boundary_coordinates = (
            polygon_features[-1]["geometry"]["coordinates"][0]
            if polygon_features
            else None
        )

    if st.session_state.drone_boundary_coordinates:

        st.caption(
            f"Boundary vertices: "
            f"{len(st.session_state.drone_boundary_coordinates) - 1}"
        )

    else:

        st.caption(
            "Use the polygon tool on the map to trace one parcel boundary."
        )

    if st.button(
        "📏 Calculate Land Area",
        use_container_width=True
    ):

        try:

            coordinates = st.session_state.drone_boundary_coordinates

            if not coordinates:

                st.error(
                    "Draw a parcel boundary on the map before calculating."
                )

                st.stop()

            polygon = Polygon(
                coordinates
            )

            if not polygon.is_valid:

                st.error(
                    "Invalid polygon: "
                    +
                    explain_validity(
                        polygon
                    )
                )

                st.stop()

            centroid = (
                polygon.centroid
            )

            utm_crs = get_utm_crs(
                centroid.y,
                centroid.x
            )

            area_m2, hectares, acres, perimeter = (
                calculate_polygon_measurements(
                    polygon,
                    utm_crs
                )
            )

            sides = (
                calculate_side_lengths(
                    polygon,
                    utm_crs
                )
            )

            st.success(
                f"Automatic metric CRS: {utm_crs}"
            )

            r1, r2, r3, r4 = (
                st.columns(4)
            )

            r1.metric(
                "Area",
                f"{area_m2:,.2f} m²"
            )

            r2.metric(
                "Hectares",
                f"{hectares:.4f} ha"
            )

            r3.metric(
                "Acres",
                f"{acres:.4f} ac"
            )

            r4.metric(
                "Perimeter",
                f"{perimeter:,.2f} m"
            )

            side_df = pd.DataFrame({

                "Side":
                    [
                        f"Side {i + 1}"
                        for i in range(
                            len(sides)
                        )
                    ],

                "Length (m)":
                    [
                        round(
                            side,
                            3
                        )
                        for side in sides
                    ]
            })

            st.subheader(
                "📏 Individual Side Lengths"
            )

            st.dataframe(
                side_df,
                use_container_width=True,
                hide_index=True
            )

            coordinate_df = pd.DataFrame({

                "Point":
                    [
                        f"P{i + 1}"
                        for i in range(
                            len(
                                coordinates
                            ) - 1
                        )
                    ],

                "Longitude":
                    [
                        c[0]
                        for c in coordinates[:-1]
                    ],

                "Latitude":
                    [
                        c[1]
                        for c in coordinates[:-1]
                    ]
            })

            st.subheader(
                "📍 Boundary Coordinates"
            )

            st.dataframe(
                coordinate_df,
                use_container_width=True,
                hide_index=True
            )

            # GeoJSON
            geojson = {

                "type":
                    "FeatureCollection",

                "features": [

                    {
                        "type":
                            "Feature",

                        "geometry":
                            {
                                "type":
                                    "Polygon",

                                "coordinates":
                                    [coordinates]
                            },

                        "properties":
                            {
                                "area_m2":
                                    area_m2,

                                "hectares":
                                    hectares,

                                "acres":
                                    acres,

                                "perimeter_m":
                                    perimeter,

                                "utm_crs":
                                    utm_crs
                            }
                    }
                ]
            }

            st.download_button(
                "⬇️ Download Boundary GeoJSON",
                json.dumps(
                    geojson,
                    indent=2
                ),
                "drone_land_boundary.geojson",
                "application/geo+json",
                use_container_width=True
            )

            # CSV
            st.download_button(
                "⬇️ Download Coordinates CSV",
                coordinate_df.to_csv(
                    index=False
                ),
                "boundary_coordinates.csv",
                "text/csv",
                use_container_width=True
            )

            # PDF
            if REPORTLAB_AVAILABLE:

                pdf_buffer = io.BytesIO()

                document = SimpleDocTemplate(
                    pdf_buffer,
                    pagesize=A4
                )

                styles = (
                    getSampleStyleSheet()
                )

                story = [

                    Paragraph(
                        "<b>GEOVISION AI "
                        "DRONE LAND SURVEY</b>",
                        styles["Title"]
                    ),

                    Spacer(
                        1,
                        20
                    )
                ]

                data = [

                    ["Parameter", "Value"],

                    [
                        "Area",
                        f"{area_m2:.2f} m²"
                    ],

                    [
                        "Hectares",
                        f"{hectares:.4f}"
                    ],

                    [
                        "Acres",
                        f"{acres:.4f}"
                    ],

                    [
                        "Perimeter",
                        f"{perimeter:.2f} m"
                    ],

                    [
                        "CRS",
                        utm_crs
                    ],

                    [
                        "Boundary Points",
                        len(coordinates) - 1
                    ]
                ]

                table = Table(
                    data
                )

                table.setStyle(
                    TableStyle([

                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.lightgrey
                        ),

                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            1,
                            colors.black
                        )
                    ])
                )

                story.append(
                    table
                )

                document.build(
                    story
                )

                st.download_button(
                    "📄 Download PDF Survey Report",
                    pdf_buffer.getvalue(),
                    "drone_land_survey_report.pdf",
                    "application/pdf",
                    use_container_width=True
                )

            st.warning(
                "This measurement is suitable for prototype/"
                "decision-support use. Legal cadastral surveying "
                "requires appropriate survey control and "
                "authorized validation."
            )

        except Exception as e:

            st.error(
                f"Measurement error: {e}"
            )


# ============================================================
# CADASTRAL IMPORT
# ============================================================

elif page == "Cadastral Import":

    st.title(
        "🗺️ Cadastral Map Import & Processing"
    )

    st.info(
        """
        Supported:
        SHP
        GeoJSON
        KML
        GPKG
        ZIP containing Shapefile
        """
    )

    uploaded = st.file_uploader(
        "Upload cadastral dataset",
        type=[
            "shp",
            "geojson",
            "json",
            "kml",
            "gpkg",
            "zip"
        ],
        key="cadastral_upload"
    )

    if uploaded:

        if st.button(
            "🚀 Import & Process",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Processing cadastral data..."
                ):

                    gdf, source, temp = (
                        load_cadastral_dataset(
                            uploaded.name,
                            uploaded.getvalue()
                        )
                    )

                    processed, stats = (
                        process_cadastral_gdf(
                            gdf
                        )
                    )

                    st.session_state.cadastral_gdf = (
                        processed
                    )

                    st.session_state.cadastral_source = (
                        source
                    )

                    st.session_state.cadastral_stats = (
                        stats
                    )

                st.success(
                    "Cadastral dataset imported."
                )

            except Exception as e:

                st.error(
                    f"Import failed: {e}"
                )

    gdf = (
        st.session_state.cadastral_gdf
    )

    stats = (
        st.session_state.cadastral_stats
    )

    if gdf is None:

        st.info(
            "No cadastral dataset loaded."
        )

        st.stop()

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Parcels",
        stats["total"]
    )

    c2.metric(
        "Min Area",
        f'{stats["min_area"]:.2f} m²'
    )

    c3.metric(
        "Max Area",
        f'{stats["max_area"]:.2f} m²'
    )

    c4.metric(
        "Average Area",
        f'{stats["avg_area"]:.2f} m²'
    )

    st.write(
        "**Source CRS:**",
        stats["source_crs"]
    )

    st.write(
        "**Area CRS:**",
        stats["area_crs"]
    )

    st.subheader(
        "📋 Parcel Attributes"
    )

    attributes = [
        c
        for c in gdf.columns
        if c != "geometry"
    ]

    st.dataframe(
        gdf[attributes].head(100),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "🗺️ Parcel Map"
    )

    if PYDECK_AVAILABLE:

        map_data = []

        for index, row in (
            gdf.head(1000).iterrows()
        ):

            try:

                geometry = row.geometry

                if geometry.geom_type == (
                    "Polygon"
                ):

                    coords = [
                        [
                            list(point)
                            for point in geometry.exterior.coords
                        ]
                    ]

                    map_data.append({

                        "polygon":
                            coords[0],

                        "area":
                            float(
                                row["area_m2"]
                            )
                    })

            except Exception:

                continue

        if map_data:

            layer = pdk.Layer(
                "PolygonLayer",
                map_data,
                get_polygon="polygon",
                get_fill_color=[
                    30,
                    120,
                    240,
                    80
                ],
                get_line_color=[
                    20,
                    60,
                    100,
                    220
                ],
                line_width_min_pixels=1,
                pickable=True
            )

            centroid = (
                gdf.to_crs(
                    4326
                ).geometry
                .unary_union
                .centroid
            )

            deck = pdk.Deck(

                layers=[layer],

                initial_view_state=pdk.ViewState(

                    latitude=centroid.y,

                    longitude=centroid.x,

                    zoom=15
                ),

                tooltip={
                    "html":
                        "Area: {area} m²"
                }
            )

            st.pydeck_chart(
                deck,
                use_container_width=True
            )

    geojson_bytes = (
        gdf.to_json()
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Processed GeoJSON",
        geojson_bytes,
        "processed_cadastral_map.geojson",
        "application/geo+json",
        use_container_width=True
    )


# ============================================================
# SPATIAL COMPARISON
# ============================================================

elif page == "Spatial Comparison":

    st.title(
        "🔍 AI vs Cadastral Comparison"
    )

    st.info(
        """
        This compares AI candidate parcel boundaries
        with cadastral polygons using spatial intersection
        and IoU.
        """
    )

    if st.session_state.cadastral_gdf is None:

        st.warning(
            "Import cadastral data first."
        )

        st.stop()

    if not st.session_state.parcels:

        st.warning(
            "No AI parcels available."
        )

        st.stop()

    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "AI Parcels",
        len(
            st.session_state.parcels
        )
    )

    c2.metric(
        "Cadastral Parcels",
        len(
            st.session_state.cadastral_gdf
        )
    )

    c3.metric(
        "GeoTIFF",
        "Loaded"
        if st.session_state.drone_raster is not None
        else "Not Loaded"
    )

    if st.button(
        "🚀 Run Spatial Comparison",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Comparing parcel boundaries..."
            ):

                results = (
                    compare_ai_with_cadastral()
                )

                st.session_state.comparison_results = (
                    results
                )

            st.success(
                "Comparison completed."
            )

        except Exception as e:

            st.error(
                f"Comparison failed: {e}"
            )

    results = (
        st.session_state.comparison_results
    )

    if results is not None:

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        if not results.empty:

            avg_iou = (
                results["IoU (%)"].mean()
            )

            mismatch = int(
                (
                    results["Status"]
                    ==
                    "Mismatch / review"
                ).sum()
            )

            c1, c2 = (
                st.columns(2)
            )

            c1.metric(
                "Average IoU",
                f"{avg_iou:.2f}%"
            )

            c2.metric(
                "Needs Review",
                mismatch
            )

            st.download_button(
                "⬇️ Download Comparison CSV",
                results.to_csv(
                    index=False
                ),
                "ai_cadastral_comparison.csv",
                "text/csv",
                use_container_width=True
            )

    st.divider()

    st.markdown(
        """
        **Prototype interpretation**

        - IoU ≥ 70% → Strong overlap
        - IoU 30–69.99% → Partial overlap
        - IoU < 30% → Mismatch / review

        These are prototype thresholds and should not be
        interpreted as legal cadastral acceptance criteria.
        """
    )


# ============================================================
# AI ANALYSIS
# ============================================================

elif page == "AI Analysis":

    st.title(
        "🤖 AI Feature Extraction"
    )

    if not st.session_state.project:

        st.warning(
            "Load or create a project first."
        )

        st.stop()

    st.info(
        """
        Current prototype mode uses deterministic demo
        parcel/building/road data.

        A production version can replace this stage with:
        YOLO / SAM / U-Net / Mask R-CNN / SegFormer
        or another trained geospatial segmentation model.
        """
    )

    st.write(
        "Project:",
        st.session_state.project["name"]
    )

    if st.button(
        "🚀 RUN AI EXTRACTION",
        use_container_width=True
    ):

        progress = st.progress(0)

        stages = [

            "Preparing imagery",

            "Creating image tiles",

            "Running segmentation",

            "Detecting buildings",

            "Detecting roads",

            "Extracting candidate boundaries",

            "Reconstructing parcels",

            "Topology validation",

            "Confidence scoring",

            "Saving GIS results"
        ]

        for i, stage in enumerate(
            stages
        ):

            progress.progress(
                int(
                    (
                        i + 1
                    )
                    /
                    len(stages)
                    *
                    100
                )
            )

            st.write(
                f"✓ {stage}"
            )

            time.sleep(
                0.15
            )

        if not st.session_state.parcels:

            load_demo()

        st.session_state.analysis_done = True

        st.success(
            "AI parcel extraction completed."
        )

    st.subheader(
        "AI Pipeline"
    )

    st.code(
        """
IMAGE
  ↓
TILING
  ↓
AI SEGMENTATION
  ↓
BUILDINGS / ROADS
  ↓
BOUNDARY EXTRACTION
  ↓
PARCEL RECONSTRUCTION
  ↓
TOPOLOGY QA
  ↓
CONFIDENCE SCORE
  ↓
HUMAN REVIEW
        """
    )


# ============================================================
# PARCEL REVIEW
# ============================================================

elif page == "Parcel Review":

    st.title(
        "🗺️ Parcel Review"
    )

    if not st.session_state.parcels:

        st.warning(
            "Load Demo Project or run AI analysis."
        )

        st.stop()

    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    if PYDECK_AVAILABLE:

        polygon_data = []

        for parcel in (
            st.session_state.parcels
        ):

            polygon_data.append({

                "polygon":
                    parcel["coordinates"],

                "parcel_id":
                    parcel["parcel_id"],

                "confidence":
                    parcel["confidence"],

                "status":
                    parcel["status"]
            })

        parcel_layer = pdk.Layer(

            "PolygonLayer",

            polygon_data,

            get_polygon="polygon",

            get_fill_color=[
                40,
                150,
                240,
                100
            ],

            get_line_color=[
                20,
                30,
                40,
                220
            ],

            line_width_min_pixels=2,

            pickable=True,

            auto_highlight=True
        )

        building_data = []

        for building in (
            st.session_state.buildings
        ):

            building_data.append({

                "polygon":
                    building["coordinates"],

                "building_id":
                    building["building_id"]
            })

        building_layer = pdk.Layer(

            "PolygonLayer",

            building_data,

            get_polygon="polygon",

            get_fill_color=[
                80,
                80,
                80,
                180
            ],

            get_line_color=[
                20,
                20,
                20,
                255
            ],

            pickable=True
        )

        road_data = []

        for road in (
            st.session_state.roads
        ):

            road_data.append({

                "path":
                    road["coordinates"],

                "road_id":
                    road["road_id"]
            })

        road_layer = pdk.Layer(

            "PathLayer",

            road_data,

            get_path="path",

            get_color=[
                255,
                255,
                255,
                255
            ],

            width_min_pixels=4
        )

        deck = pdk.Deck(

            layers=[
                road_layer,
                building_layer,
                parcel_layer
            ],

            initial_view_state=pdk.ViewState(

                latitude=16.504,

                longitude=78.103,

                zoom=14
            ),

            tooltip={
                "html":
                    """
                    <b>{parcel_id}</b><br>
                    Confidence: {confidence}%<br>
                    Status: {status}
                    """
            }
        )

        st.pydeck_chart(
            deck,
            use_container_width=True
        )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    table = pd.DataFrame([

        {

            "Parcel":
                p["parcel_id"],

            "Confidence":
                p["confidence"],

            "Bucket":
                p["bucket"],

            "Area m²":
                p["area"],

            "Perimeter m":
                p["perimeter"],

            "Status":
                p["status"],

            "QA":
                ", ".join(
                    p["qa"]
                )
        }

        for p in (
            st.session_state.parcels
        )
    ])

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # SELECT PARCEL
    # --------------------------------------------------------

    selected_id = st.selectbox(

        "Select Parcel",

        [
            p["parcel_id"]
            for p in (
                st.session_state.parcels
            )
        ]
    )

    selected = find_parcel(
        selected_id
    )

    if selected:

        st.session_state.selected_parcel = (
            selected_id
        )

        st.subheader(
            f"Parcel {selected_id}"
        )

        c1, c2, c3 = (
            st.columns(3)
        )

        c1.metric(
            "Confidence",
            f'{selected["confidence"]}%'
        )

        c2.metric(
            "Area",
            f'{selected["area"]:.2f} m²'
        )

        c3.metric(
            "Perimeter",
            f'{selected["perimeter"]:.2f} m'
        )

        st.markdown(
            confidence_badge(
                selected["bucket"]
            ),
            unsafe_allow_html=True
        )

        st.write(
            "QA:",
            ", ".join(
                selected["qa"]
            )
        )

        st.write(
            "Status:",
            selected["status"]
        )

        st.divider()

        a1, a2, a3, a4 = (
            st.columns(4)
        )

        with a1:

            if st.button(
                "✅ Accept",
                use_container_width=True
            ):

                selected["status"] = (
                    "Accepted"
                )

                add_history(
                    selected_id,
                    "Accepted"
                )

                st.success(
                    "Parcel accepted."
                )

                st.rerun()

        with a2:

            if st.button(
                "✏️ Edit",
                use_container_width=True
            ):

                st.session_state.editing = True

        with a3:

            if st.button(
                "❌ Reject",
                use_container_width=True
            ):

                selected["status"] = (
                    "Rejected"
                )

                add_history(
                    selected_id,
                    "Rejected"
                )

                st.error(
                    "Parcel rejected."
                )

                st.rerun()

        with a4:

            if st.button(
                "📍 Field Verify",
                use_container_width=True
            ):

                selected["status"] = (
                    "Field Verification"
                )

                add_history(
                    selected_id,
                    "Sent to field verification"
                )

                st.warning(
                    "Sent for field verification."
                )

                st.rerun()

        # ----------------------------------------------------
        # EDIT
        # ----------------------------------------------------

        if st.session_state.editing:

            st.subheader(
                "Edit Boundary"
            )

            coords_text = st.text_area(

                "Polygon coordinates",

                json.dumps(
                    selected["coordinates"],
                    indent=2
                ),

                height=220
            )

            if st.button(
                "💾 Save Boundary"
            ):

                try:

                    new_coordinates = (
                        json.loads(
                            coords_text
                        )
                    )

                    polygon = Polygon(
                        new_coordinates
                    )

                    if not polygon.is_valid:

                        st.error(
                            explain_validity(
                                polygon
                            )
                        )

                    else:

                        selected[
                            "coordinates"
                        ] = (
                            new_coordinates
                        )

                        selected[
                            "status"
                        ] = "Edited"

                        add_history(
                            selected_id,
                            "Boundary edited"
                        )

                        st.session_state.editing = (
                            False
                        )

                        st.success(
                            "Boundary updated."
                        )

                        st.rerun()

                except Exception:

                    st.error(
                        "Invalid polygon coordinates."
                    )


# ============================================================
# FIELD CHECK
# ============================================================

elif page == "Field Check":

    st.title(
        "📍 Field Verification"
    )

    field_parcels = [

        p

        for p in (
            st.session_state.parcels
        )

        if (
            p["confidence"] < 60
            or
            p["status"]
            ==
            "Field Verification"
        )
    ]

    if not field_parcels:

        st.success(
            "No parcels require field verification."
        )

    for parcel in field_parcels:

        with st.container(
            border=True
        ):

            st.subheader(
                parcel["parcel_id"]
            )

            st.write(
                f'Confidence: '
                f'{parcel["confidence"]}%'
            )

            st.write(
                "Issues:",
                ", ".join(
                    parcel["qa"]
                )
            )

            if st.button(
                f'✓ Verify {parcel["parcel_id"]}'
            ):

                parcel["status"] = (
                    "Field Verified"
                )

                add_history(
                    parcel["parcel_id"],
                    "Field verification completed"
                )

                st.success(
                    "Field verification completed."
                )

                st.rerun()


# ============================================================
# EXPORTS
# ============================================================

elif page == "Exports":

    st.title(
        "📦 Export Results"
    )

    if not st.session_state.parcels:

        st.warning(
            "No parcel results available."
        )

        st.stop()

    features = []

    for parcel in (
        st.session_state.parcels
    ):

        features.append({

            "type":
                "Feature",

            "geometry":
                {
                    "type":
                        "Polygon",

                    "coordinates":
                        [
                            parcel[
                                "coordinates"
                            ]
                        ]
                },

            "properties":
                {
                    "parcel_id":
                        parcel["parcel_id"],

                    "area":
                        parcel["area"],

                    "perimeter":
                        parcel["perimeter"],

                    "confidence":
                        parcel["confidence"],

                    "bucket":
                        parcel["bucket"],

                    "status":
                        parcel["status"],

                    "qa":
                        parcel["qa"]
                }
        })

    geojson = {

        "type":
            "FeatureCollection",

        "features":
            features
    }

    st.download_button(

        "⬇️ Download Parcel GeoJSON",

        json.dumps(
            geojson,
            indent=2
        ),

        "geovision_parcels.geojson",

        "application/geo+json",

        use_container_width=True
    )

    csv_df = pd.DataFrame([

        {

            "parcel_id":
                p["parcel_id"],

            "area_m2":
                p["area"],

            "perimeter_m":
                p["perimeter"],

            "confidence":
                p["confidence"],

            "bucket":
                p["bucket"],

            "status":
                p["status"],

            "qa":
                "; ".join(
                    p["qa"]
                )
        }

        for p in (
            st.session_state.parcels
        )
    ])

    st.download_button(

        "⬇️ Download Parcel CSV",

        csv_df.to_csv(
            index=False
        ),

        "geovision_parcels.csv",

        "text/csv",

        use_container_width=True
    )

    # --------------------------------------------------------
    # PROJECT REPORT
    # --------------------------------------------------------

    report = f"""
GEOVISION AI
URBAN PARCEL MAPPING REPORT

Project:
{st.session_state.project}

TOTAL PARCELS:
{len(st.session_state.parcels)}

HIGH CONFIDENCE:
{len([p for p in st.session_state.parcels if p["confidence"] >= 85])}

INSPECT:
{len([p for p in st.session_state.parcels if 60 <= p["confidence"] < 85])}

FIELD CHECK:
{len([p for p in st.session_state.parcels if p["confidence"] < 60])}

BUILDINGS:
{len(st.session_state.buildings)}

ROADS:
{len(st.session_state.roads)}

REVIEW EVENTS:
{len(st.session_state.history)}

DISCLAIMER:
AI candidate parcel boundaries are not legal cadastral
boundaries. Final decisions require authoritative records
and appropriate professional validation.
"""

    st.download_button(

        "📄 Download Project Report",

        report,

        "geovision_project_report.txt",

        "text/plain",

        use_container_width=True
    )

    if st.session_state.comparison_results is not None:

        st.download_button(

            "⬇️ Download AI-Cadastral Comparison",

            st.session_state.comparison_results.to_csv(
                index=False
            ),

            "ai_cadastral_comparison.csv",

            "text/csv",

            use_container_width=True
        )


# ============================================================
# ARCHITECTURE
# ============================================================

elif page == "Architecture":

    st.title(
        "🏗️ System Architecture"
    )

    st.code(
        """
    GEO VISION AI | AI-ASSISTED URBAN PARCEL MAPPING

    [DRONE RGB] + [DSM / DTM] + [EXISTING GIS LAYERS]
                        |
                        v
    [DATA PREPROCESSING: georeference, orthomosaic, denoise,
                     CRS transform, raster processing]
                        |
                        v
    [AI / ML ENGINE: PyTorch segmentation and detection]
                        |
                +----------+----------+
                v          v          v
            [BUILDINGS]   [ROADS]  [LAND USE]
                +----------+----------+
                        v
    [PARCEL BOUNDARY EXTRACTION: preliminary polygons]
                        |
                        v
    [GIS PROCESSING: GeoPandas | GDAL / Rasterio |
                  Shapely | PyProj]
                        |
                        v
    [TOPOLOGY VALIDATION: geometry | overlaps | gaps |
                      boundary consistency]
                        |
                        v
    [CONFIDENCE SCORING: HIGH | MEDIUM | LOW]
                +----------+----------+
                |                     |
          HIGH CONFIDENCE       MEDIUM / LOW
                |                     |
                |           [SURVEYOR VERIFICATION]
                |           [ACCEPT | EDIT | REJECT |
                |              FIELD VERIFY]
                +----------+----------+
                        v
        [PYTHON / FASTAPI API: projects | imagery | jobs |
                  parcels | review | exports]
                    |
                    v
    [POSTGRESQL + POSTGIS: parcels | buildings | roads |
                        land use | confidence]
                        |
                        v
    [WEB-GIS: React.js + Vite on Node.js | MapLibre GL JS]
                        |
                        v
    [FINAL OUTPUT: cadastral map | parcel boundaries |
                 GIS layers | survey-ready exports]
        """
    )

    st.markdown(
        "**Web stack:** React.js frontend · Node.js toolchain · Vite · MapLibre GL JS"
    )

    st.subheader(
        "Operational Workflow"
    )

    st.code(
        """
START
    |
    v
Upload drone imagery + DSM / DTM + GIS layers
    |
    v
Georeference and preprocess imagery
    |
    v
AI feature extraction: buildings / roads / land use
    |
    v
Extract parcel boundaries and generate polygons
    |
    v
Run GIS topology checks: valid? overlaps? gaps?
    | INVALID                    | VALID
    v                            v
Correct geometry / flag issues -> Calculate confidence
                                                 |
                                +--------+--------+
                                |                 |
                            HIGH            MEDIUM / LOW
                                |                 v
                                |        Surveyor verification
                                |        Accept / Edit / Reject /
                                |        Field verify
                                +--------+--------+
                                                 v
                                    Store in PostGIS
                                                 |
                                                 v
                                     Display in Web-GIS
                                                 |
                                                 v
                                 Final cadastral map
                                                 |
                                                END
        """
    )

    st.markdown(
        "**PPT workflow:** Drone Imagery + DSM/DTM + GIS Data → Preprocessing → "
        "AI Feature Extraction → Parcel Boundary Detection → Cadastral Polygon "
        "Generation → GIS Topology Validation → Confidence Scoring → Surveyor "
        "Verification → PostGIS → Web-GIS → Final Cadastral Map"
    )

    st.info(
        "Prototype status: this Streamlit app runs GIS checks and deterministic "
        "demo extraction. PyTorch inference and PostgreSQL/PostGIS persistence "
        "are production targets; the React + FastAPI implementation is available "
        "in the project workspace."
    )


# ============================================================
# SETTINGS
# ============================================================

elif page == "Settings":

    st.title(
        "⚙️ Settings"
    )

    st.subheader(
        "AI Configuration"
    )

    ai_mode = st.selectbox(
        "AI Model",
        [
            "Prototype / Stub",
            "YOLO",
            "SAM",
            "U-Net",
            "SegFormer"
        ]
    )

    st.write(
        "Selected:",
        ai_mode
    )

    st.subheader(
        "Confidence Thresholds"
    )

    high_threshold = st.slider(
        "High Confidence",
        50,
        100,
        85
    )

    inspect_threshold = st.slider(
        "Inspection Threshold",
        0,
        90,
        60
    )

    st.write(
        f"""
        High Confidence:
        {high_threshold}–100%

        Inspect:
        {inspect_threshold}–{high_threshold - 1}%

        Field Check:
        < {inspect_threshold}%
        """
    )

    st.subheader(
        "Default CRS"
    )

    st.text_input(
        "CRS",
        "EPSG:4326"
    )

    st.info(
        "For metric land calculations, the application "
        "automatically selects a local UTM CRS."
    )


# ============================================================
# REVIEW HISTORY
# ============================================================

if page in [
    "Parcel Review",
    "Field Check"
]:

    st.divider()

    st.subheader(
        "🕒 Review History"
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No review history yet."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        font-size:0.8rem;
        padding:10px;
    ">
    GeoVision AI — AI-Assisted Urban Parcel Mapping,
    Cadastral Feature Extraction & Drone Land Survey
    <br>
    Candidate boundaries are not legal cadastral boundaries.
    </div>
    """,
    unsafe_allow_html=True
)