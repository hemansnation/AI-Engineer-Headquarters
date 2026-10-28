TABLES = {
    'customers': {'about': 'People who bought from Lakeview. Use it for counts by city or signup date.', 'columns': {'id': 'customer id', 'city': 'city the customer lives in', 'created_at': 'signup date, YYYY-MM-DD'}, 'hidden': ['full_name', 'email', 'phone']},
    'products': {'about': 'Appliances Lakeview sells, with price and warranty length.', 'columns': {'id': 'product id', 'name': 'product name', 'category': 'laundry, cooling or kitchen', 'price_inr': 'list price in rupees', 'warranty_months': 'warranty length in months'}, 'hidden': []},
    'orders': {'about': 'One row per order. Use it for order status, order dates and revenue.', 'columns': {'id': 'order id', 'customer_id': 'joins to customers.id', 'order_date': 'date of order, YYYY-MM-DD', 'status': 'delivered, shipped, processing or cancelled', 'total_inr': 'order value in rupees'}, 'hidden': []},
    'order_items': {'about': 'Products inside each order. Use it to connect orders to products.', 'columns': {'id': 'line id', 'order_id': 'joins to orders.id', 'product_id': 'joins to products.id', 'quantity': 'units bought', 'unit_price_inr': 'price per unit in rupees'}, 'hidden': []},
    'warranty_claims': {'about': 'Warranty claims raised by customers on a product from an order.', 'columns': {'id': 'claim id', 'order_id': 'joins to orders.id', 'product_id': 'joins to products.id', 'opened_on': 'date the claim was opened, YYYY-MM-DD', 'status': 'open, approved, rejected or closed', 'issue': 'short description of the fault'}, 'hidden': []},
}
ALLOWED_TABLES = set(TABLES)
HIDDEN_COLUMNS = {column for card in TABLES.values() for column in card['hidden']}
def card_text(table):
    card = TABLES[table]
    columns = ', '.join(name + ' (' + meaning + ')' for name, meaning in card['columns'].items())
    return 'Table ' + table + ': ' + card['about'] + ' Columns: ' + columns + '.'
def all_cards():
    return {table: card_text(table) for table in TABLES}
