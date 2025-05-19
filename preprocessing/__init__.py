import os
import sys

# Set the preprocessing directory as its own root for imports and file access
preprocessing_dir = os.path.dirname(os.path.abspath(__file__))
if preprocessing_dir not in sys.path:
    sys.path.insert(0, preprocessing_dir)

META_TAGS = {
    "religious" : ["pope", "god", "jesus", "bible", "church", "devil", "angel", "heaven", "hell", "satan", "jesus christ", "spiritual", "faith",
                   "demon", "religion"],
    "nsfw" : ["gore", "nudity", "sex", "group sex", "violence", "explicit sex", "gruesome", "erection", "nudity (full frontal - notable)",
              "breasts", "nudity (topless - notable)", "nudity (topless)"],
    "childrens" : ["disney", "disney animated feature", "animated"],
    "oscar_winner" : ["oscar (best supporting actor)" ,"oscar (best actor)","oscar (best directing)","oscar winner: best picture","oscar (best picture)",
                      "oscar (best supporting actress)","oscar (best actress)"],
    "oscar_nominee" : ["oscar nominee: best picture","oscar nominee: best actor","oscar nominee: best actress","oscar nominee: best supporting actor",
                      "oscar nominee: best supporting actress","oscar nominee: best director"],
    "notable" : ["afi 100", "imdb top 250", "national film registry"],
    "genres" : [
        "action", "adventure", "animation", "children's", "comedy", "crime", "documentary", "drama", "fantasy", "film-noir", "horror", "musical", "mystery", "romance", "sci-fi", "thriller", "war", "western", "(no genres listed)"
    ]
}

# Optionally, set as working directory for relative file access
# os.chdir(preprocessing_dir)
