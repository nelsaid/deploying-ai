

# Assignment 2 — HealthCompass
By Nouar ElSaid 

## Overview

HealthCompass is a conversational AI system that provides structured access to World Bank health and economic indicators through an interactive chat interface. Instead of navigating raw World Bank datasets or complex dashboards, users can simply ask questions in natural language. HealthCompass translates those questions into structured data queries and returns clean, readable summaries.

HealthCompass allows users to:

*View country-level health and economic indicators*
Example: Japan gdp - Canada life expectancy

*Compare countries across key metrics*
Example: compare UK vs Japan gini - compare USA vs Germany health spending 

*Retrieve historical data for specific years*
Example: India gdp 2010 - Brazil life expectancy 2005

*Explore inequality, mortality, health spending, and environmental exposure*
Example: South Africa gini - Nigeria adult mortality -Indonesia pm2.5

*Ask for clear definitions of technical indicators*
Example:  What is gini? -  Explain NCD mortality - Define health spending per capit

The system combines deterministic routing, semantic search, and function calling to ensure accurate and predictable responses while maintaining conversational flexibility.

System Components:

* Deterministic rule-based routing
* Semantic search using persistent embeddings
* Function-calling via OpenAI API
* A Gradio chat interface with conversational memory

The assistant has a professional personality which returns structured data to answer the specific request. If the request is out of scope the chatbot redirects the user to request an indicator, specify a country or compare countries. 

## Services

### Service 1 — World Bank API (Structured Data Service)

This service retrieves health and economic indicators from the World Bank API.

The supported indicators are:

* Life expectancy
* NCD mortality
* Adult mortality
* Health spending per capita
* Physicians per 1,000
* GDP per capita
* Gini Inequality index
* PM2.5 air pollution

Through HealthCompass the API output is transformed into structured natural-language summaries and is never returned verbatim.


### Service 2 — Semantic Query via Persistent ChromaDB

This service handles definition-style queries such as:

* “What is gini?” | * “Explain life expectancy.” | "Define NCD mortality" 

A locally persisted ChromaDB instance stores indicator definitions. Queries are resolved using semantic similarity search.
Embedding generation is performed offline and included in this repository’s documentation.


### Service 3 — Function Calling Router

The third service uses OpenAI function-calling to:

* Route user requests to appropriate handlers
* Interpret comparisons
* Interpret structured commands

Deterministic grammar enforcement overrides model outputs when necessary to ensure predictable behavior.

## User Interface

The system uses Gradio to provide a chat-based interface.

### Personality

The assistant:

* Is clear and concise
* Highlights meaningful data trends - what is the lowest value or the highest value and if that's the favourable or not.
* Avoids jargon
* Uses flags and structured formatting for readability

### Memory

The system maintains conversational memory via:

* A persistent `default_country` setting; users can set a country as a default and explore its statistics. To exit the user types "clear" or "exit" 
* Short-term message history passed into the LLM router

## Guardrails

The system includes guardrails that:

* Prevent revealing or modifying the system prompt
* Any requests that are out of scope are redirected
* Block restricted topics:
  * Cats or dogs
  * Horoscopes or zodiac signs
  * Taylor Swift

## Design Decisions

* Deterministic routing is prioritized over LLM inference whenever possible to keep the output predictable.
* Multi-turn clarification memory was intentionally simplified to reduce instability.
* Indicator-only queries either use the default country (if set) or return all-country snapshots.


## Limitations

* All comparison requests must include the indicator in the same message.
* Only the set of supported countries and indicators are accepted.
* The system does not perform long-term memory persistence beyond session memory.
* There is room for improving the chat interface to provide the user a better expierence ( for example, if a user omits something the chatbot can follow up, suggets indicators, etc. )

