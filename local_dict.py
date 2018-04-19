synonyms = {
	"school bus": set(["school bus", "bus", "vehicle", "commercial vehicle", "transport"])
}

def valid_label(label, true_label):
    return label in synonyms[true_label]

