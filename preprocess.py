"""
Galaxy Zoo Preprocessing Pipeline

Statistical Modelling of Galaxy Morphology
Final Project — Statistical Inference and Modelling
December 2025

Authors: Tina Sikharulidze, Albert Lamb, Michael Duarte Gonçalves

This module prepares SDSS photometric data for morphological classification.
We correct for Galactic extinction, compute color indices and structural
parameters, and apply standard cleaning procedures.
"""

import numpy as np
import pandas as pd


# Columns we don't need for morphology modelling
UNWANTED_COLS = [
    "PETROR50_R_KPC_SIMPLE_BIN", "PETROMAG_MR_SIMPLE_BIN",
    "REDSHIFT_SIMPLE_BIN", "WVT_BIN",
    "ROWC_U", "COLC_U", "ROWC_G", "COLC_G", "ROWC_R", "COLC_R",
    "ROWC_I", "COLC_I", "ROWC_Z", "COLC_Z",
    "RUN", "RERUN", "CAMCOL", "FIELD", "OBJ",
    "RA", "DEC", "REGION",
]

# Magnitude-extinction pairs for dust correction
EXTINCTION_PAIRS = [
    ("PETROMAG_U", "EXTINCTION_U"),
    ("PETROMAG_G", "EXTINCTION_G"),
    ("PETROMAG_R", "EXTINCTION_R"),
    ("PETROMAG_I", "EXTINCTION_I"),
    ("PETROMAG_Z", "EXTINCTION_Z"),
]

# Radius columns (log-transformed without clipping)
RADII_COLS = ["PETROR50_R", "PETROR90_R", "PETROR50_R_KPC"]

# Error columns (log-transformed with clipping, use 99999 as sentinel)
ERROR_COLS = [
    "PETROMAGERR_U", "PETROMAGERR_G", "PETROMAGERR_R",
    "PETROMAGERR_I", "PETROMAGERR_Z",
    "PETROMAGERR_MU", "PETROMAGERR_MG", "PETROMAGERR_MR",
    "PETROMAGERR_MI", "PETROMAGERR_MZ",
    "DEVMAGERR_R", "EXPMAGERR_R", "CMODELMAGERR_R",
]

# All columns to log-transform
FLUX_LIKE_COLS = RADII_COLS + ERROR_COLS


class GalaxyZooPreprocessor:
    """Preprocessing pipeline for SDSS Galaxy Zoo photometry."""

    def preprocess(self, df):
        """Run the full pipeline. Returns a cleaned DataFrame."""
        df = df.copy()

        df = self._drop_columns(df, UNWANTED_COLS)
        df = self._correct_extinction(df)
        df = self._fix_error_encoding(df)

        df = self._add_colors(df)
        df = self._add_concentration(df)
        df = self._add_surface_brightness(df)
        df = self._log_transform(df)

        df = df.dropna()
        df = self._remove_outliers(df)
        df = self._drop_columns(df, ["OBJID"])

        return df

    # ---- Cleaning steps ----

    def _drop_columns(self, df, cols):
        """Drop columns if they exist."""
        to_drop = [c for c in cols if c in df.columns]
        return df.drop(columns=to_drop)

    def _correct_extinction(self, df):
        """Apply Galactic extinction correction to Petrosian magnitudes."""
        df = df.copy()

        for mag, ext in EXTINCTION_PAIRS:
            if mag in df.columns and ext in df.columns:
                df[mag + "_corr"] = df[mag] - df[ext]

        # Always drop all magnitude and extinction columns
        all_cols = [mag for mag, _ in EXTINCTION_PAIRS] + [ext for _, ext in EXTINCTION_PAIRS]
        return df.drop(columns=[c for c in all_cols if c in df.columns])

    def _fix_error_encoding(self, df):
        """Replace 99999 sentinel values with NaN in error columns."""
        for col in ERROR_COLS:
            if col in df.columns:
                df[col] = df[col].replace(99999.0, np.nan)
        return df

    # ---- Feature engineering ----

    def _add_colors(self, df):
        """Compute adjacent-band color indices (e.g., u-g, g-r)."""
        bands = ["PETROMAG_U", "PETROMAG_G", "PETROMAG_R", "PETROMAG_I", "PETROMAG_Z"]

        for i in range(len(bands) - 1):
            b1, b2 = bands[i], bands[i + 1]
            if b1 in df.columns and b2 in df.columns:
                df[f"{b1}_{b2}_color"] = df[b1] - df[b2]

        return df

    def _add_concentration(self, df):
        """Compute concentration index C = R90/R50."""
        if "PETROR90_R" in df.columns and "PETROR50_R" in df.columns:
            r50 = df["PETROR50_R"].replace(0, np.nan)
            df["CONC_R"] = df["PETROR90_R"] / r50
        return df

    def _add_surface_brightness(self, df):
        """Compute mean surface brightness within R50."""
        if "PETROMAG_R" in df.columns and "PETROR50_R" in df.columns:
            r50 = df["PETROR50_R"].replace(0, np.nan)
            df["SURFACE_BRIGHTNESS_R"] = df["PETROMAG_R"] + 2.5 * np.log10(2 * np.pi * r50**2)
        return df

    def _log_transform(self, df):
        """Log-transform radii and errors, then drop originals."""
        df = df.copy()

        # Radii: no clipping (original behavior)
        for col in RADII_COLS:
            if col in df.columns:
                df[f"LOG_{col}"] = np.log1p(df[col])

        # Errors: clip to zero before log (original behavior)
        for col in ERROR_COLS:
            if col in df.columns:
                df[f"LOG_{col}"] = np.log1p(df[col].clip(lower=0))

        return df.drop(columns=[c for c in FLUX_LIKE_COLS if c in df.columns])

    # ---- Outlier removal ----

    def _remove_outliers(self, df, threshold=10.0):
        """Remove extreme outliers using IQR method (threshold * IQR)."""
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        
        # Don't apply outlier detection to ID column
        if "OBJID" in numeric_cols:
            numeric_cols.remove("OBJID")

        mask = pd.Series(False, index=df.index)
        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            mask |= (df[col] < q1 - threshold * iqr) | (df[col] > q3 + threshold * iqr)

        return df[~mask]
