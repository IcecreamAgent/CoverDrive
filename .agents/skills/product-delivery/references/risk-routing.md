# Risk routing

Use the highest applicable level and add orthogonal tags such as `ux`, `data`, `security`, `release`, or `external-ai`.

| Level | Typical change | Minimum route |
| --- | --- | --- |
| L0 | Copy, low-impact style, narrow mechanical fix | Main Agent implements and performs targeted verification |
| L1 | Ordinary feature or bug within an established pattern | Explore as needed, implement, test, and independently review meaningful risk |
| L2 | New or materially changed user journey | Define an experience contract, resolve material UX choices, implement, run product checks, and preserve human/user evidence where required |
| L3 | Authentication, payment, sensitive data, migrations, shared contracts, or release | Architecture/risk review, isolated execution where useful, dedicated checks, rollback preparation, and environment-specific authorization |

Risk level answers what must be verified. `autonomy` answers which actions are allowed:

- `analysis`: inspect, reproduce, test, and propose; do not change product implementation.
- `implementation`: modify the assigned workspace and prepare a validated candidate.
- `delivery`: may also merge, deploy, or roll back only within separately recorded authorization.

## UX contract

For a new or uncertain interaction, record the representative user task, entry point, success endpoint, applicable states, failure recovery, target devices, open question, and validation method before implementation.

Automated browser or component checks can prove observable behavior. Agent heuristics can identify risks. Only real participant records count as user testing, and only actual product data counts as an outcome signal. Minor UI fixes can use proportionate Agent and automated checks when no material usability hypothesis changes.

## Human gates

Escalate when the next safe action needs a new product choice, material architecture choice, external account or environment, expanded scope, irreversible data operation, or production authorization. Present the evidence, recommended option, alternatives, impact, and what remains possible without the decision.

