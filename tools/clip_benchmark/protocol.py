"""Frozen native CLIP/SigLIP2 text pairs and weight revisions; no GPU imports."""

PROMPTS = {
    "fight": ["Two people are physically fighting, punching or kicking each other.",
              "People are walking, standing or talking peacefully."],
    "smoking": ["A person is smoking a cigarette.",
                "A person is eating food or drinking from a cup."],
    "fall": ["A person is losing balance and falling to the ground.",
             "A person is standing, walking or sitting steadily."],
    "weapon": ["A person is holding a gun or a knife.",
               "A person is holding an ordinary object such as a phone."],
    "bag_knife": ["A knife is visibly exposed inside a bag or being taken out of a bag.",
                  "A bag contains ordinary items with no visible knife."],
    "bag_alcohol_bottle": [
        "A bottle recognizable as an alcohol bottle is visibly exposed inside a bag or being taken out of a bag.",
        "A bag contains ordinary items with no visible alcohol bottle.",
    ],
}
REVISIONS = {
    "openai/clip-vit-large-patch14": "32bd64288804d66eefd0ccbe215aa642df71cc41",
    "google/siglip2-base-patch16-224": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
}
