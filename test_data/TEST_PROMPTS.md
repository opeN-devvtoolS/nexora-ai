# Natural-language tests for AgentOS

The six CSV files here are synthetic and unrelated. Each has 220 rows, some missing feature values, and an ID column. The questions below sound like ordinary user requests. They deliberately do not tell the agent which tool, algorithm, or exact column list to use.

## How to run

1. Copy one CSV into this project's workspace/input. Keep the other CSVs in this test_data folder.
2. Open a terminal in this project folder and run: py -3 agent.py
3. Ask the seven questions for that CSV in order. Exit and restart the agent for the next CSV.
4. The first prediction for a new filename should train a model if none is saved. Later predictions should reuse it. If you have already tested that filename, the agent may find its saved model immediately.
5. Check that answers use the file's actual numbers, the model uses the intended target and sensible features, the ID column is not used to predict, both candidate models are evaluated, and predictions come from tools. The questions do not promise exact prediction values.

These CSVs do not specify a currency. If the agent adds dollars or another currency, that is an unsupported assumption. A high R² or low RMSE is a test result, not a guarantee for any one new case. RAG is not part of these tests.

## 1. Aircraft inspections — aircraft_faults.csv

The question is whether a component develops a fault within 30 days. Inspection ID is only a row identifier.

1. "I added aircraft_faults.csv. What kind of information is in it, and is anything missing?"
2. "How many inspections ended up with a fault within a month, and how many didn't?"
3. "Do the A, B, and C inspection grades have different average vibration readings?"
4. "Are flight hours, component age, vibration, or temperature related to each other in this file? Check the data."
5. "I've got a component with about 8,000 flight hours, 80 months of age, vibration 6.8 mm/s, temperature 50 C, and inspection grade C. Does the file suggest it might fault in the next 30 days? Train a model if you need to; the inspection ID shouldn't matter."
6. "Which model did you use for that answer? How did it and the other model do when tested on unseen rows?"
7. "What about a component with only 500 flight hours, 5 months of age, vibration 0.8 mm/s, temperature 5 C, and grade A? Use the model you already trained and compare the two predictions."

## 2. Apartment rent — apartment_rent.csv

The target is monthly rent. Listing ID is only a row identifier. The file does not state a currency.

1. "What's in this apartment file? How many listings are there, and is any information missing?"
2. "What's the average monthly rent in each neighborhood? Which one is highest?"
3. "Show me the three most expensive apartments and their size, bedrooms, and neighborhood."
4. "Do larger apartments generally have higher rent in this data? Check the numbers rather than guessing."
5. "Roughly what rent would you expect for a 1,500 sq ft, three-bedroom place in Central? It's 10 years old, 5 km from the center, and has parking. Use the data, and train a model if there isn't one yet."
6. "How good was the model on unseen apartments? Did the other model do better or worse?"
7. "What if it were only 600 sq ft, one bedroom, 35 years old, 25 km away, in Suburban, with no parking? Use the saved model and compare with the earlier place. Is either estimate outside the rents seen in the file?"

## 3. Streaming subscriptions — subscription_churn.csv

The question is whether an account cancels next month. Account ID is only a row identifier. The monthly fee has no stated currency.

1. "What's in subscription_churn.csv, and where are the gaps in the data?"
2. "How many accounts cancelled the next month versus stayed?"
3. "Which type of plan has the highest average monthly fee?"
4. "Do people who watch more tend to contact support less? Use the numeric data to check."
5. "A Basic subscriber joined two months ago, pays 45 a month, watches about one hour a week, has contacted support five times, and doesn't use autopay. Do you think they'll cancel next month? Use this file to train a model if needed; account ID isn't useful here."
6. "What model did you choose for that prediction, and how did both candidate models perform?"
7. "Now try someone who's subscribed for 50 months, pays 15, watches 30 hours a week, has no support tickets, uses Premium, and has autopay. Use the saved model and compare the actual predictions."

## 4. Crop yield — crop_yield.csv

The target is yield in tons per hectare. Field ID is only a row identifier.

1. "What does crop_yield.csv contain? Are any measurements missing?"
2. "Which soil type has the highest average yield in this file?"
3. "Show me the three fields with the greatest yield. What were their rainfall and fertilizer amounts?"
4. "Is yield related to rainfall or fertilizer in this data? Check the correlations, but don't call that proof of cause."
5. "A Loamy field gets 850 mm of rainfall, averages 25 C, receives 150 kg of fertilizer per hectare, and has irrigation. What yield would you estimate? Use the CSV and train a model if needed; ignore the field ID."
6. "How well did the chosen model predict held-out fields? What were the other model's results?"
7. "Compare that with a Sandy field with 250 mm of rain, 38 C average temperature, 20 kg of fertilizer per hectare, and no irrigation. Use the saved model for both values."

## 5. Parcel deliveries — parcel_delivery.csv

The question is whether a parcel arrives late. Parcel ID is only a row identifier.

1. "What's recorded in parcel_delivery.csv? Is any of the information missing?"
2. "How many parcels were late, and how many were not?"
3. "What's the average delivery distance for each weather condition?"
4. "Show me the three longest-distance parcels. What were their weather and late-delivery results?"
5. "A non-priority parcel is going 2,200 km in snowy weather. The warehouse is at 95% load and the courier has one year of experience. Does it look likely to arrive late? Use the file to train if necessary; don't use parcel ID."
6. "Which model made that prediction? Compare it with the other model using cross-validation and test results."
7. "Now what about a priority parcel going 50 km in clear weather, warehouse load 45%, with a courier who has 12 years of experience? Reuse the saved model and compare the answers."

## 6. Solar energy — solar_output.csv

The target is daily energy output in kWh. Site ID is only a row identifier.

1. "What can you tell me about solar_output.csv? How many records are there, and what values are missing?"
2. "Which panel type has the highest average daily energy output?"
3. "Show me the three highest-output sites. How much sun did they get and what was their panel capacity?"
4. "In this file, how are sunshine, cloud cover, panel capacity, and energy output related? Check the numeric correlations."
5. "I have a one-year-old, 10 kW Monocrystalline panel. If it gets 10 hours of sunshine with 10% cloud cover, about how much energy would it produce in a day? Train a model from the file if needed; site ID shouldn't be used."
6. "Which model did you use, and how did both models perform on cross-validation and the final test rows?"
7. "What would the saved model predict for an 18-year-old, 3 kW Thin Film panel with 2 hours of sunshine and 90% cloud cover? Compare the two predictions without retraining."
