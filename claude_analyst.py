"""
Claude API integration for contextual analysis and natural language reports.
"""

import json
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-20250514"


def claude_analyze_matchup(home_team: str, away_team: str,
                            home_form: dict, away_form: dict, league: str) -> dict:
    """Ask Claude to evaluate contextual factors. Returns JSON scores."""
    prompt = f"""You are an expert football match analyst. Analyze the upcoming match
and return ONLY valid JSON (no markdown, no comments) with the following scores 0.0 to 1.0:

Match: {home_team} (home) vs {away_team} (away) | League: {league}

{home_team} last 5 matches:
- Avg goals scored: {home_form.get('avg_GF', 0):.2f}
- Avg goals conceded: {home_form.get('avg_GA', 0):.2f}
- Avg shots on target: {home_form.get('avg_SoT', 0):.2f}
- Form (avg points): {home_form.get('Form', 0):.2f}

{away_team} last 5 matches:
- Avg goals scored: {away_form.get('avg_GF', 0):.2f}
- Avg goals conceded: {away_form.get('avg_GA', 0):.2f}
- Avg shots on target: {away_form.get('avg_SoT', 0):.2f}
- Form (avg points): {away_form.get('Form', 0):.2f}

Return JSON in this exact format:
{{
    "home_attack_strength": <float>,
    "home_defense_strength": <float>,
    "away_attack_strength": <float>,
    "away_defense_strength": <float>,
    "home_momentum": <float>,
    "away_momentum": <float>,
    "match_intensity_prediction": <float>,
    "upset_probability": <float>,
    "home_win_confidence": <float>,
    "draw_likelihood": <float>,
    "reasoning": "<1-2 sentence explanation>"
}}"""

    message = client.messages.create(
        model=MODEL, max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        return {}


def claude_analyze_divergence(match: str, bookmaker: dict, polymarket: dict,
                               ml_model: dict, poly_liquidity: float, poly_volume_24h: float) -> str:
    """Claude analyzes divergences between three probability sources."""
    prompt = f"""You are a senior sports analyst. Analyze these three probability sources for a football match.

**Match:** {match}

| Source | Home | Draw | Away |
|---|---|---|---|
| Bookmaker (Bet365) | {bookmaker['home']:.1%} | {bookmaker['draw']:.1%} | {bookmaker['away']:.1%} |
| Polymarket | {polymarket['home']:.1%} | {polymarket['draw']:.1%} | {polymarket['away']:.1%} |
| ML Model | {ml_model['home']:.1%} | {ml_model['draw']:.1%} | {ml_model['away']:.1%} |

Polymarket: Liquidity=${poly_liquidity:,.0f} | 24h Volume=${poly_volume_24h:,.0f}

Tasks:
1. Where are the main divergences and what might they mean?
2. Which source is more reliable here and why?
3. Signs of informed trading on Polymarket? (unusual volume/price shift)
4. Final prediction with confidence level.

Be concise, 5-8 sentences."""

    message = client.messages.create(
        model=MODEL, max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def generate_prediction_report(home_team: str, away_team: str,
                                model_proba: dict, stats: dict, league: str) -> str:
    """Generate a full analytical report for one match."""
    prompt = f"""You are a professional football analyst. Write a concise analytical report.

Match: **{home_team}** vs **{away_team}** ({league})

ML Ensemble probabilities:
- {home_team} win: {model_proba['home_win']:.1%}
- Draw: {model_proba['draw']:.1%}
- {away_team} win: {model_proba['away_win']:.1%}

{home_team} (last 5): Goals scored {stats.get('home_avg_GF', 0):.2f} | Conceded {stats.get('home_avg_GA', 0):.2f} | SoT {stats.get('home_avg_SoT', 0):.1f} | Form {stats.get('home_Form', 0):.2f}
{away_team} (last 5): Goals scored {stats.get('away_avg_GF', 0):.2f} | Conceded {stats.get('away_avg_GA', 0):.2f} | SoT {stats.get('away_avg_SoT', 0):.1f} | Form {stats.get('away_Form', 0):.2f}

Include: key factors, team strengths/weaknesses, prediction, confidence (high/medium/low), upset scenarios."""

    message = client.messages.create(
        model=MODEL, max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def analyze_matchday(matches: list) -> str:
    """Batch analyze a full matchday in one Claude call."""
    matches_text = ""
    for i, m in enumerate(matches, 1):
        matches_text += (
            f"\n{i}. {m['home']} vs {m['away']}\n"
            f"   ML: H={m['prob_H']:.0%} D={m['prob_D']:.0%} A={m['prob_A']:.0%}\n"
            f"   Home form: {m.get('home_form', 0):.2f} | Away form: {m.get('away_form', 0):.2f}\n"
        )

    prompt = f"""Analyze this matchday. For each match provide: prediction (1/X/2), confidence (⭐/⭐⭐/⭐⭐⭐), brief comment.

{matches_text}

End with the 1-2 best picks of the day."""

    message = client.messages.create(
        model=MODEL, max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
