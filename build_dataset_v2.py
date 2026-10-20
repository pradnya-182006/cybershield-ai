"""CyberShield AI - Phase 2B Dataset Builder.

Generates data/synthetic_comments_v2.json (160 comments) by preserving all
original 100 comments from data/synthetic_comments.json and adding 60 curated
comments with enhanced threat diversity, difficult negative contrast examples,
idiomatic expressions, and separate benign vs hostile sarcasm.
"""

import json
import os

with open("data/synthetic_comments.json", "r", encoding="utf-8") as f:
    ORIGINAL_100 = json.load(f)

NEW_60 = [
    # =========================================================================
    # 1. EXPANDED THREAT EXAMPLES (SYN101 - SYN116: 16 comments)
    # Direct, conditional, stalking, intimidation, and implied violent harm
    # =========================================================================
    {
        "id": "SYN101",
        "platform": "synthetic",
        "comment": "I am going to put you in the hospital if you ever show your face here again.",
        "labels": ["toxic", "threat"]
    },
    {
        "id": "SYN102",
        "platform": "synthetic",
        "comment": "You will pay for this with your life, just wait and see what happens.",
        "labels": ["threat"]
    },
    {
        "id": "SYN103",
        "platform": "synthetic",
        "comment": "I will hunt you down and destroy you for what you did to my family.",
        "labels": ["toxic", "threat"]
    },
    {
        "id": "SYN104",
        "platform": "synthetic",
        "comment": "Keep talking like that and I promise you will leave this room in an ambulance.",
        "labels": ["threat"]
    },
    {
        "id": "SYN105",
        "platform": "synthetic",
        "comment": "If you do not take down that video right now, I will come to your house and force you.",
        "labels": ["threat"]
    },
    {
        "id": "SYN106",
        "platform": "synthetic",
        "comment": "Say that to my face in person and see how fast I break your jaw.",
        "labels": ["toxic", "threat"]
    },
    {
        "id": "SYN107",
        "platform": "synthetic",
        "comment": "If you show up at the meetup tomorrow, you won't walk out on your own feet.",
        "labels": ["threat"]
    },
    {
        "id": "SYN108",
        "platform": "synthetic",
        "comment": "If I catch you on campus after dark, you are going to regret opening your mouth.",
        "labels": ["threat"]
    },
    {
        "id": "SYN109",
        "platform": "synthetic",
        "comment": "I have photos of your front door and I know what time you get home from work.",
        "labels": ["threat"]
    },
    {
        "id": "SYN110",
        "platform": "synthetic",
        "comment": "I am standing outside your building right now, come out and face what is coming.",
        "labels": ["threat"]
    },
    {
        "id": "SYN111",
        "platform": "synthetic",
        "comment": "You better sleep with one eye open tonight, you won't hear me coming.",
        "labels": ["threat"]
    },
    {
        "id": "SYN112",
        "platform": "synthetic",
        "comment": "I know your daily routine and where you park your car every single morning.",
        "labels": ["threat"]
    },
    {
        "id": "SYN113",
        "platform": "synthetic",
        "comment": "People who cross me usually end up needing emergency surgery, remember that.",
        "labels": ["threat"]
    },
    {
        "id": "SYN114",
        "platform": "synthetic",
        "comment": "Fatal accidents happen all the time to people who talk too much online.",
        "labels": ["threat"]
    },
    {
        "id": "SYN115",
        "platform": "synthetic",
        "comment": "You are playing with fire and someone is going to get seriously hurt very soon.",
        "labels": ["threat"]
    },
    {
        "id": "SYN116",
        "platform": "synthetic",
        "comment": "Keep pushing people and see how quickly your luck runs out in real life.",
        "labels": ["threat"]
    },

    # =========================================================================
    # 2. DIFFICULT NEGATIVE EXAMPLES FOR THREAT (SYN117 - SYN130: 14 comments)
    # Idioms, reporting harm, prevention, gaming, fiction, medical contexts
    # All are safe!
    # =========================================================================
    {
        "id": "SYN117",
        "platform": "synthetic",
        "comment": "That intense leg workout yesterday is going to kill my muscles all week.",
        "labels": ["safe"]
    },
    {
        "id": "SYN118",
        "platform": "synthetic",
        "comment": "My laptop battery is dying, let me grab my charger before the call drops.",
        "labels": ["safe"]
    },
    {
        "id": "SYN119",
        "platform": "synthetic",
        "comment": "That stand-up comedy special was hilarious, I almost died laughing.",
        "labels": ["safe"]
    },
    {
        "id": "SYN120",
        "platform": "synthetic",
        "comment": "Our engineering team needs to execute this deployment plan and eliminate system bugs.",
        "labels": ["safe"]
    },
    {
        "id": "SYN121",
        "platform": "synthetic",
        "comment": "I am completely starving after that hike, I could eat a horse right now.",
        "labels": ["safe"]
    },
    {
        "id": "SYN122",
        "platform": "synthetic",
        "comment": "Someone sent a threat to my younger brother online, how do I report this account to moderators?",
        "labels": ["safe"]
    },
    {
        "id": "SYN123",
        "platform": "synthetic",
        "comment": "We must take all threats of cyberbullying seriously and support affected students.",
        "labels": ["safe"]
    },
    {
        "id": "SYN124",
        "platform": "synthetic",
        "comment": "The security team intervened quickly and prevented anyone from being hurt at the venue.",
        "labels": ["safe"]
    },
    {
        "id": "SYN125",
        "platform": "synthetic",
        "comment": "I am writing a research paper on how social media platforms detect violent threats.",
        "labels": ["safe"]
    },
    {
        "id": "SYN126",
        "platform": "synthetic",
        "comment": "In this video game boss fight, your character has to dodge incoming attacks or you will die.",
        "labels": ["safe"]
    },
    {
        "id": "SYN127",
        "platform": "synthetic",
        "comment": "The villain in the movie threatened to destroy the entire solar system.",
        "labels": ["safe"]
    },
    {
        "id": "SYN128",
        "platform": "synthetic",
        "comment": "Our football defense completely crushed the opposing team in the fourth quarter.",
        "labels": ["safe"]
    },
    {
        "id": "SYN129",
        "platform": "synthetic",
        "comment": "The doctor said my grandfather needs to stay in the hospital for two more days of observation.",
        "labels": ["safe"]
    },
    {
        "id": "SYN130",
        "platform": "synthetic",
        "comment": "I am reading a classic thriller novel about a detective tracking down a dangerous suspect.",
        "labels": ["safe"]
    },

    # =========================================================================
    # 3. AGGRESSIVE / TOXIC NON-THREATS (SYN131 - SYN138: 8 comments)
    # Hostile language that is toxic or insulting, but definitely NOT a threat
    # =========================================================================
    {
        "id": "SYN131",
        "platform": "synthetic",
        "comment": "I completely despise your toxic attitude, get off my feed and stay away.",
        "labels": ["toxic"]
    },
    {
        "id": "SYN132",
        "platform": "synthetic",
        "comment": "You are the most arrogant, dishonest person I have ever encountered on this forum.",
        "labels": ["toxic", "insult"]
    },
    {
        "id": "SYN133",
        "platform": "synthetic",
        "comment": "Nobody asked for your ridiculous opinion, shut up and go away forever.",
        "labels": ["toxic", "insult"]
    },
    {
        "id": "SYN134",
        "platform": "synthetic",
        "comment": "You are an incompetent fraud who ruins every collaborative project you touch.",
        "labels": ["toxic", "insult"]
    },
    {
        "id": "SYN135",
        "platform": "synthetic",
        "comment": "Stop spamming your garbage conspiracy theories, you sound completely unhinged.",
        "labels": ["toxic"]
    },
    {
        "id": "SYN136",
        "platform": "synthetic",
        "comment": "I hate everything about this terrible update, what a shameful disaster of a release.",
        "labels": ["toxic"]
    },
    {
        "id": "SYN137",
        "platform": "synthetic",
        "comment": "You are a selfish, manipulative hypocrite and everyone secretly knows it.",
        "labels": ["toxic", "insult"]
    },
    {
        "id": "SYN138",
        "platform": "synthetic",
        "comment": "Delete this atrocity of a post right now, you are thoroughly embarrassing yourself.",
        "labels": ["toxic"]
    },

    # =========================================================================
    # 4. SARCASM & OBSCENE EXPANSION (SYN139 - SYN150: 12 comments)
    # Benign irony vs hostile sarcasm vs vulgar complaints
    # =========================================================================
    {
        "id": "SYN139",
        "platform": "synthetic",
        "comment": "Oh wonderful, another three-hour flight delay, exactly how I wanted to spend my Friday evening.",
        "labels": ["sarcasm"]
    },
    {
        "id": "SYN140",
        "platform": "synthetic",
        "comment": "Sure, because who does not love when software updates automatically restart your PC during work?",
        "labels": ["sarcasm"]
    },
    {
        "id": "SYN141",
        "platform": "synthetic",
        "comment": "Great job weather forecast, you said zero percent chance of rain right before the thunderstorm.",
        "labels": ["sarcasm"]
    },
    {
        "id": "SYN142",
        "platform": "synthetic",
        "comment": "Oh look, Einstein decided to grace us with another completely useless observation.",
        "labels": ["toxic", "insult", "sarcasm"]
    },
    {
        "id": "SYN143",
        "platform": "synthetic",
        "comment": "You must have worked really hard to come up with an explanation that makes zero sense.",
        "labels": ["toxic", "sarcasm"]
    },
    {
        "id": "SYN144",
        "platform": "synthetic",
        "comment": "Bravo on another masterclass in how to completely fail at basic communication.",
        "labels": ["toxic", "insult", "sarcasm"]
    },
    {
        "id": "SYN145",
        "platform": "synthetic",
        "comment": "What a load of horse shit, this company has the worst customer support ever.",
        "labels": ["obscene"]
    },
    {
        "id": "SYN146",
        "platform": "synthetic",
        "comment": "Holy crap, that was an extraordinarily close call on the highway this morning.",
        "labels": ["obscene"]
    },
    {
        "id": "SYN147",
        "platform": "synthetic",
        "comment": "Stop talking out of your ass when you clearly have no idea what happened.",
        "labels": ["toxic", "obscene"]
    },
    {
        "id": "SYN148",
        "platform": "synthetic",
        "comment": "This goddamn printer jams every single time I need to print urgent documents.",
        "labels": ["obscene"]
    },
    {
        "id": "SYN149",
        "platform": "synthetic",
        "comment": "Bullshit corporate excuses won't fix the fact that the entire database went down.",
        "labels": ["obscene"]
    },
    {
        "id": "SYN150",
        "platform": "synthetic",
        "comment": "Fuck this noise, I am canceling my monthly subscription right now.",
        "labels": ["obscene"]
    },

    # =========================================================================
    # 5. CONSTRUCTIVE DEBATE & SENSITIVE DIALOGUE (SYN151 - SYN160: 10 comments)
    # Dialect, civil disagreement, respectful debate - all safe!
    # =========================================================================
    {
        "id": "SYN151",
        "platform": "synthetic",
        "comment": "We need an open and honest discussion about mental health resources available on campus.",
        "labels": ["safe"]
    },
    {
        "id": "SYN152",
        "platform": "synthetic",
        "comment": "I respectfully disagree with the proposed tax policy, here are the economic data points.",
        "labels": ["safe"]
    },
    {
        "id": "SYN153",
        "platform": "synthetic",
        "comment": "Discrimination against any minority community has no place in our organization.",
        "labels": ["safe"]
    },
    {
        "id": "SYN154",
        "platform": "synthetic",
        "comment": "Could someone please explain the historical context behind this regional conflict?",
        "labels": ["safe"]
    },
    {
        "id": "SYN155",
        "platform": "synthetic",
        "comment": "I appreciate you sharing your lived experience with disability accommodations.",
        "labels": ["safe"]
    },
    {
        "id": "SYN156",
        "platform": "synthetic",
        "comment": "Let us focus our critique on the argument itself rather than attacking the speaker.",
        "labels": ["safe"]
    },
    {
        "id": "SYN157",
        "platform": "synthetic",
        "comment": "The documentary highlights systemic inequality in the criminal justice system.",
        "labels": ["safe"]
    },
    {
        "id": "SYN158",
        "platform": "synthetic",
        "comment": "That is an interesting cultural tradition, thank you for teaching me about it.",
        "labels": ["safe"]
    },
    {
        "id": "SYN159",
        "platform": "synthetic",
        "comment": "Our debate yesterday was intense, but I value that we remained civil throughout.",
        "labels": ["safe"]
    },
    {
        "id": "SYN160",
        "platform": "synthetic",
        "comment": "I am grateful for this supportive community, it has helped me through a tough time.",
        "labels": ["safe"]
    }
]

if __name__ == "__main__":
    combined = ORIGINAL_100 + NEW_60
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "synthetic_comments_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(combined)} synthetic comments to {out_path}")
