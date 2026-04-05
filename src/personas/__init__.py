"""Trading personas -- advisory research agents with independent mandates.

Each persona is a mini research analyst with its own strategy, ticker
universe, notional budget, and Claude API mandate.  Personas detect
market conditions, generate research briefs with trade recommendations,
and present them to the user for manual action.

Usage::

    from src.personas.engine import PersonaEngine
    engine = PersonaEngine(config)
    engine.check_events()           # detect conditions for dormant personas
    brief = engine.generate_brief("core")  # produce advisory brief
"""
