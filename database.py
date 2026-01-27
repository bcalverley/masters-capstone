from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY


def lookup_card(prediction):
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    set_code, card_number_str = prediction.split()
    card_number = int(card_number_str)

    response = supabase.table("card_instances").select(
        "set_code, card_number, name, rarity"
    ).eq("set_code", set_code).eq("card_number", card_number).execute()

    return response.data[0] if response.data else None
