from huggingface_hub import login
login() #ζητάει access token
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Φορτώνω το μοντέλο και τον tokenizer από το αποθηκευμένο μονοπάτι
model_path = "/content/greek-sentiment-final-model"  


model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ανέβασμα του μοντέλου
model.push_to_hub("DimiPatsi/greek-movie-sentiment-bert")

# Ανέβασμα του tokenizer
tokenizer.push_to_hub("DimiPatsi/greek-movie-sentiment-bert")

print("Το μοντέλο και ο tokenizer ανέβηκαν επιτυχώς!")
