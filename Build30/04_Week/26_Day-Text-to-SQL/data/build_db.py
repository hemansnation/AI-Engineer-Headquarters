import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path
DB = Path('data/lakeview.db')
CITIES = ['Pune', 'Indore', 'Jaipur', 'Nagpur', 'Bhopal', 'Surat']
PRODUCTS = [('Washing machine 7kg', 'laundry', 24999, 24), ('Washing machine 9kg', 'laundry', 32999, 24), ('Dryer', 'laundry', 27999, 12), ('Refrigerator 260L', 'cooling', 28999, 36), ('Refrigerator 450L', 'cooling', 54999, 36), ('Split AC 1.5 ton', 'cooling', 41999, 12), ('Microwave oven', 'kitchen', 9999, 12), ('Dishwasher', 'kitchen', 38999, 24), ('Water purifier', 'kitchen', 14999, 12), ('Chimney', 'kitchen', 17999, 24)]
STATUSES = ['delivered'] * 7 + ['shipped', 'processing', 'cancelled']
CLAIM_STATUSES = ['open', 'approved', 'rejected', 'closed']
def build(path=DB, seed=26):
    random.seed(seed)
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript('''
    CREATE TABLE customers (id INTEGER PRIMARY KEY, full_name TEXT, email TEXT, phone TEXT, city TEXT, created_at TEXT);
    CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price_inr INTEGER, warranty_months INTEGER);
    CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), order_date TEXT, status TEXT, total_inr INTEGER);
    CREATE TABLE order_items (id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders(id), product_id INTEGER REFERENCES products(id), quantity INTEGER, unit_price_inr INTEGER);
    CREATE TABLE warranty_claims (id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders(id), product_id INTEGER REFERENCES products(id), opened_on TEXT, status TEXT, issue TEXT);
    CREATE TABLE staff_salaries (id INTEGER PRIMARY KEY, staff_name TEXT, monthly_salary_inr INTEGER);
    ''')
    for i in range(1, 201):
        joined = date(2025, 1, 1) + timedelta(days=random.randint(0, 600))
        con.execute('INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)', (i, 'Customer ' + str(i), 'customer' + str(i) + '@example.com', '+91-90000' + str(10000 + i), random.choice(CITIES), joined.isoformat()))
    for i, (name, category, price, months) in enumerate(PRODUCTS, start=1):
        con.execute('INSERT INTO products VALUES (?, ?, ?, ?)', (i, name, category, price, months))
    item_id = 0
    for order_id in range(1, 1501):
        day = date(2026, 1, 1) + timedelta(days=random.randint(0, 289))
        lines = []
        for _ in range(random.choice([1, 1, 1, 2])):
            product_id = random.randint(1, len(PRODUCTS))
            lines.append((product_id, random.choice([1, 1, 2]), PRODUCTS[product_id - 1][2]))
        total = sum(q * p for _, q, p in lines)
        con.execute('INSERT INTO orders VALUES (?, ?, ?, ?, ?)', (order_id, random.randint(1, 200), day.isoformat(), random.choice(STATUSES), total))
        for product_id, quantity, price in lines:
            item_id += 1
            con.execute('INSERT INTO order_items VALUES (?, ?, ?, ?, ?)', (item_id, order_id, product_id, quantity, price))
    issues = ['drum noise', 'not cooling', 'water leak', 'display error', 'door seal', 'motor failure']
    for claim_id in range(1, 121):
        order_id = random.randint(1, 1500)
        product_id = con.execute('SELECT product_id FROM order_items WHERE order_id = ? LIMIT 1', (order_id,)).fetchone()[0]
        opened = date(2026, 2, 1) + timedelta(days=random.randint(0, 260))
        con.execute('INSERT INTO warranty_claims VALUES (?, ?, ?, ?, ?, ?)', (claim_id, order_id, product_id, opened.isoformat(), random.choice(CLAIM_STATUSES), random.choice(issues)))
    for i in range(1, 13):
        con.execute('INSERT INTO staff_salaries VALUES (?, ?, ?)', (i, 'Agent ' + str(i), random.randint(18000, 32000)))
    con.commit()
    con.close()
    return path
if __name__ == '__main__':
    print('Database written to ' + str(build()))
