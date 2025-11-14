# API URL

```
https://web-production-1e855.up.railway.app/ask
```


# Bonus 1
1. Pre-filter Using a local name classifier. For a RAG architecture, the rule is deterministic filtering -> LLM Model for reasoning and final output. Classifier examples are: Simple keyword matching where it filters for the member's name, fuzzy matching to handle typos and grammtical errors, and Regex Name Matching. This approach would be cheaper because only 1 call would be made to OpenAI to produce the final output, not 2. This would also speed up response time and avoid hallunications.

2. Full knowledge graph representation. This would include a CRON job to still fetch every hour and update new information as nodes. People and messages can be nodes and topics would be edges. This would be extremely accurate for detailed queries where relationships can be traversed and can scale well. No external LLM calls needed. This architecture is very complex and would require set-up to an external DB such as Neo4j.  


# Bonus 2
1. Member data has incomplete and incoherent messages that don't specify a request or accommodation. Such an example would be:
    {
      "id": "23cdd9fb-535b-4416-a400-5107a2a5cd08",
      "user_id": "cd3a350e-dbd2-408f-afa0-16a072f56d23",
      "user_name": "Sophia Al-Farsi",
      "timestamp": "2024-11-18T00:06:40.175781+00:00",
      "message": "I finally"
    }


2. Messages will be 'Thank You' comments or blanket statements rather than all requests. This can be costly since there would have to be an additional filter to sort messages with important context vs irrelevant context.

3. Message don't include important details needed to make a request or booking which can hinder the user's schedule or cause additional delays. An example is: "I need four front-row seats for the game on November 20."

4. Messages are not fully grammtically correct which can hinder semantic meaning inside the vector DB. An example is:
   {
      "id": "c49ab4ea-c417-4348-ac4d-d538edcc565e",
      "user_id": "5b2e7346-eef5-445d-a063-6c5267f04bf8",
      "user_name": "Hans Müller",
      "timestamp": "2024-12-24T08:13:22.160298+00:00",
      "message": "I’m flying to San Francisco—book the first class for two on November 10."
    }


