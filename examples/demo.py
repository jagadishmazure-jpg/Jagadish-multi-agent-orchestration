"""Research pipeline with checks, ticket triage through a router, and the iteration budget."""
from collections import Counter

from orchestrator.patterns import UnknownRouteError
from scenarios import run_research_scenario, triage_ticket

r = run_research_scenario()
rec = r["trajectory"]
print("Research pipeline agents:", " -> ".join(r["agents"]))
print("Trajectory events:", len(rec.events), dict(Counter(e.kind for e in rec.events)))
print("Final:", r["final"])
final_ok = all(k in r["final"] for k in ("Python", "Guido", "1991"))
print("Checks: keywords present:", final_ok, "| errors:", r["errors"] or "none")

for ticket in ("please refund my last invoice",
               "login error after update",
               "need to change my account email",
               "what are your opening hours?"):
    try:
        t = triage_ticket(ticket)
    except UnknownRouteError as e:
        print(f"Ticket '{ticket}' -> not routed: {e} (no default route, so it fails loudly)")
        continue
    print(f"Ticket '{ticket}' -> {t['category']} / {t['agent']}")
    print("   ", t["output"])

billing = triage_ticket("please refund my last invoice")["trajectory"]
print("Billing trajectory:", [(e.kind, e.agent) for e in billing.events])
