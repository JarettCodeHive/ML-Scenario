from cluster_c.synthetic import generate_term_deals, write_term_deals

if __name__ == "__main__":
    frame = generate_term_deals()
    write_term_deals(frame)
    print(f"Wrote {len(frame):,} deterministic sample rows")
