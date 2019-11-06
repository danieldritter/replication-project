import jsonlines


with jsonlines.open("/media/daniel/DATA/diplomacy_data/standard_no_press.jsonl") as file:
    for i,obj in enumerate(file):
        print(obj.keys())
        print(obj["phases"][0]["orders"])
        print(obj["phases"][0].keys())
        print(obj["phases"][0]["results"])

        break
