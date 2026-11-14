"""Service layer between the bot / LangGraph tools and the storage layer.

Every tool here resolves birth data from stored profiles only: the account
owner's natal profile or a ``user_contacts`` row owned by that account. That
is the security boundary of the forecast feature - arbitrary coordinates or
dates cannot enter the calculation through a request.
"""
