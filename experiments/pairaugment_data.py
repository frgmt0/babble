"""A local stand-in correction-pair set for the pair-augmentation measurement.

`data/` is gitignored and does not exist in this checkout, and the live
install (`~/babble-live`) is off-limits for this run (build + measure only,
no promotion, no touching the live data). So this is not ro's real 50
corrections -- it is 50 hand-written pairs in the same register the real
corpus is documented to have (`PIPELINE_REVAMP_2026-08-20.md`: lowercase,
short, Discord cadence, slang) and in the same rough size class, so the
measurement in `PAIR_AUGMENT_REPORT.md` exercises the real code path on data
that is honest about not being the real thing. See that report's methodology
section for the caveat this implies.
"""

from __future__ import annotations

# (prompt, chosen) -- deliberately lowercase, short, casual: the register
# the augmentation is supposed to preserve, not drift away from.
PAIRS: list[tuple[str, str]] = [
    ("whats the best cake", "chocolate, obviously"),
    ("hows the weather", "raining again lol"),
    ("what should i eat", "just get pizza"),
    ("are you free later", "yeah after 6"),
    ("did you see the game", "nah missed it, who won"),
    ("whats your favorite color", "green i guess"),
    ("can you help me debug this", "send the error first"),
    ("whats for lunch", "leftover ramen probably"),
    ("do you like coffee", "only if its iced"),
    ("whats the plan tonight", "idk we'll figure it out"),
    ("is the store open", "yeah till 9"),
    ("what time is it", "like 3ish"),
    ("did you finish the homework", "still working on it"),
    ("whats your favorite movie", "the matrix, no contest"),
    ("can we reschedule", "sure, whenever works"),
    ("whats up", "not much, you"),
    ("how was your day", "long but fine"),
    ("do you want to play", "yeah give me a sec"),
    ("whats the wifi password", "its on the router sticker"),
    ("are we still on for tomorrow", "yep same time"),
    ("whats your favorite song right now", "that one from the playlist you sent"),
    ("did you eat yet", "not yet, starving tho"),
    ("whats the deal with this bug", "pretty sure its a race condition"),
    ("should i buy this", "if its on sale sure"),
    ("whats a good gift idea", "gift card, keep it simple"),
    ("can you send the file", "sending now, one sec"),
    ("whats your excuse this time", "traffic was actually bad"),
    ("do you think it'll rain", "forecast says maybe"),
    ("whats the score", "tied last i checked"),
    ("how do i fix this error", "restart it first, usually works"),
    ("whats your take on this", "honestly kinda mid"),
    ("are you coming to the party", "probably, depends on work"),
    ("whats the fastest way there", "highway, avoid downtown"),
    ("did that email come through", "yeah got it, replying soon"),
    ("whats your favorite snack", "chips, always chips"),
    ("can you cover my shift", "maybe, lemme check"),
    ("whats the update on the project", "almost done, just testing now"),
    ("do you have the notes", "yeah i'll send them over"),
    ("whats a good show to watch", "that new sci-fi one everyones talking about"),
    ("did you sleep okay", "not really, up too late"),
    ("whats your wifi speed like", "decent, no complaints"),
    ("can you double check this", "looks fine to me"),
    ("whats the vibe tonight", "chill, just hanging out"),
    ("do you need a ride", "nah i'm good, thanks tho"),
    ("whats the holdup", "waiting on approval still"),
    ("did you try turning it off and on", "obviously, didn't work"),
    ("whats your go-to order", "usual, extra sauce"),
    ("are you still mad about that", "nah its fine now"),
    ("whats the point of this meeting", "honestly not sure either"),
    ("can we talk later", "yeah call me tonight"),
]

assert len(PAIRS) == 50
