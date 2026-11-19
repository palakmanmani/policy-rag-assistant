import time
from rag import load_pdf, chunk_pages, build_index, answer

tests = [
    ("What is the waiting period for pre-existing diseases?", ["24 months"]),
    ("Is maternity covered by default?", ["optional"]),
    ("After how many months of continuous coverage can the Maternity Cover be used, and how many deliveries does it allow?", ["36 months"]),
    ("What is the grace period to renew the policy?", ["30 days"]),
    ("How long is the free look period?", ["15 days","fifteen days"]),
    ("How early must pre-authorization be requested for a planned hospitalisation?", ["48 hours"]),
    ("How much Guaranteed Cumulative Bonus is given for a claim-free year?", ["20%"]),
    ("What is the maximum payout for a road ambulance on reimbursement?", ["10,000", "10000"]),
    ("What is the co-payment for Worldwide Cover?", ["10%"]),
    ("What is the cataract claim limit after two years?", ["1 lakh", "1,00,000", "100,000", "1 lac"]),
    ("Does this policy cover space travel?", ["could not find"]),
    ("What is the annual premium for a 30-year-old?", ["could not find"]),
    ("What is the claim settlement ratio of ICICI Lombard?", ["could not find"]),
]

pages = load_pdf("data/policy.pdf")
chunks = chunk_pages(pages)
print("Loading... please wait")
vectors = build_index(chunks)


def ask_with_retry(question):
    """Try up to 4 times. If the quota error appears, wait and try again."""
    for attempt in range(4):
        try:
            return answer(question, chunks, vectors) or "", False
        except Exception as e:
            if "429" in str(e) or "503" in str(e):
                print("  Quota limit hit. Waiting 60 seconds, then trying again...")
                time.sleep(60)
            else:
                return "ERROR: " + str(e), True
    return "ERROR: quota still exceeded", True


score = 0
errors = 0
for i, (question, keywords) in enumerate(tests, start=1):
    reply, had_error = ask_with_retry(question)
    print(f"\nQ{i}: {question}")
    if had_error:
        errors += 1
        print("Result: ERROR (not counted as pass or fail)")
    else:
        ok = any(k.lower() in reply.lower() for k in keywords)
        if ok:
            score += 1
        print("Result:", "PASS" if ok else "FAIL")
    print("Answer:", reply[:300])
    time.sleep(15)  # slow down so we stay inside the free limit

print(f"\nFinal score: {score} / {len(tests)}   (errors: {errors})")