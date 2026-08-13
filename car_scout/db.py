"""Persistance du 'garage' : historique des opportunités retenues par l'utilisateur."""

from pathlib import Path

import pandas as pd

DB_FILE = Path(__file__).resolve().parent.parent / "garage_db.csv"

COLUMNS = ["Date", "Titre", "Prix Achat", "KM", "Année", "Ville", "Score", "Statut", "Bénéfice Net"]


def load_db() -> pd.DataFrame:
    if DB_FILE.exists():
        return pd.read_csv(DB_FILE)
    return pd.DataFrame(columns=COLUMNS)


def save_entry(entry: dict) -> None:
    df = pd.concat([load_db(), pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(DB_FILE, index=False)
