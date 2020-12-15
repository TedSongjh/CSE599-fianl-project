import json
import os
with open('category.json','r') as cat:
    category = json.load(cat)
with open('object_ann.json','r') as obj:
    object_ann = json.load(obj)
new_category = [
    {"name": "human",
     "token": []},
    {"name": "cyclist",
     "token": []},
    {"name": "car",
     "token": []},
    {"name": "truck",
     "token": []},
    {"name": "others",
     "token": []},

]
for obj in category:
    if obj["name"][:5] == "human":
        new_category[0]["token"].append(obj["token"])
    elif obj["name"][:7] == "vehicle":
        if obj["name"][8:] == "motorcycle" or obj["name"][8:] == "bicycle":
            new_category[1]["token"].append(obj["token"])
        elif obj["name"][8:] == "truck" or obj["name"][8:] == "trailer":
            new_category[3]["token"].append(obj["token"])
        else:
            new_category[2]["token"].append(obj["token"])
    else:
        new_category[4]["token"].append(obj["token"])

for obj in object_ann:
    for cat in new_category:
        if obj["token"] in cat["token"]:
            obj["token"] = cat["token"][0]
            break

with open("new_category","w") as outfile:
    json.dump(new_category, outfile)
with open("new_object_ann","w") as outfile:
    json.dump(object_ann, outfile)
