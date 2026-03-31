"""
view_log.py — Browse the ESP32-CAM detection database
Usage:
    python view_log.py                  # interactive menu
    python view_log.py --export         # dump both tables to CSV
"""

import sqlite3
import argparse
import csv
import os
from datetime import datetime

DB_PATH = "detections.db"


def open_db(path):
    if not os.path.exists(path):
        print(f"[ERROR] Database not found: {path}")
        exit(1)
    return sqlite3.connect(path)


def print_table(rows, headers):
    if not rows:
        print("  (no records)")
        return
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
              for i, h in enumerate(headers)]
    sep  = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt  = "|" + "|".join(f" {{:<{w}}} " for w in widths) + "|"
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)


def menu_texts(conn):
    print("\n── TEXT DETECTIONS ──────────────────────────────")
    print("  1. All records")
    print("  2. DANGER records only")
    print("  3. Last 10 records")
    print("  4. Records containing a word")
    choice = input("Choice: ").strip()

    if choice == "1":
        rows = conn.execute(
            "SELECT id, timestamp, corrected, priority, avg_conf "
            "FROM text_detections ORDER BY id").fetchall()
        print_table(rows, ["id", "timestamp", "corrected", "priority", "conf"])

    elif choice == "2":
        rows = conn.execute(
            "SELECT id, timestamp, corrected, avg_conf "
            "FROM text_detections WHERE priority='DANGER' ORDER BY id"
        ).fetchall()
        print_table(rows, ["id", "timestamp", "text", "conf"])

    elif choice == "3":
        rows = conn.execute(
            "SELECT id, timestamp, corrected, priority "
            "FROM text_detections ORDER BY id DESC LIMIT 10"
        ).fetchall()
        print_table(rows, ["id", "timestamp", "text", "priority"])

    elif choice == "4":
        word = input("Search word: ").strip().lower()
        rows = conn.execute(
            "SELECT id, timestamp, corrected, priority "
            "FROM text_detections WHERE LOWER(corrected) LIKE ? ORDER BY id",
            (f"%{word}%",)
        ).fetchall()
        print_table(rows, ["id", "timestamp", "text", "priority"])


def menu_objects(conn):
    print("\n── OBJECT DETECTIONS ────────────────────────────")
    print("  1. All records")
    print("  2. Group by label (counts)")
    choice = input("Choice: ").strip()

    if choice == "1":
        rows = conn.execute(
            "SELECT id, timestamp, label, confidence "
            "FROM object_detections ORDER BY id"
        ).fetchall()
        print_table(rows, ["id", "timestamp", "label", "conf"])

    elif choice == "2":
        rows = conn.execute(
            "SELECT label, COUNT(*) as count, ROUND(AVG(confidence),3) "
            "FROM object_detections GROUP BY label ORDER BY count DESC"
        ).fetchall()
        print_table(rows, ["label", "count", "avg_conf"])


def export_csv(conn, path_prefix="export"):
    # Text detections
    rows = conn.execute("SELECT * FROM text_detections").fetchall()
    t_path = f"{path_prefix}_text.csv"
    with open(t_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","timestamp","raw_text","corrected",
                    "entities","priority","avg_conf"])
        w.writerows(rows)
    print(f"[EXPORTED] {t_path}  ({len(rows)} rows)")

    # Object detections
    rows = conn.execute("SELECT * FROM object_detections").fetchall()
    o_path = f"{path_prefix}_objects.csv"
    with open(o_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","timestamp","label","confidence","bbox"])
        w.writerows(rows)
    print(f"[EXPORTED] {o_path}  ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--export", action="store_true",
                        help="Export both tables to CSV and exit")
    args = parser.parse_args()

    conn = open_db(args.db)

    if args.export:
        export_csv(conn)
        return

    # Summary
    t = conn.execute("SELECT COUNT(*) FROM text_detections").fetchone()[0]
    o = conn.execute("SELECT COUNT(*) FROM object_detections").fetchone()[0]
    print(f"\n══ Detection Log Viewer ══  db={args.db}")
    print(f"   Text detections  : {t}")
    print(f"   Object detections: {o}")

    while True:
        print("\n── MENU ─────────────────────────────────────────")
        print("  t — browse text detections")
        print("  o — browse object detections")
        print("  e — export to CSV")
        print("  q — quit")
        ch = input("Choice: ").strip().lower()
        if ch == 't':
            menu_texts(conn)
        elif ch == 'o':
            menu_objects(conn)
        elif ch == 'e':
            export_csv(conn)
        elif ch == 'q':
            break

    conn.close()


if __name__ == "__main__":
    main()