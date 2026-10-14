import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Category = Literal['billing', 'delivery', 'refund', 'login', 'product_defect', 'other']
Urgency = Literal['low', 'medium', 'high']
Sentiment = Literal['positive', 'neutral', 'negative']

ORDER_ID_PATTERN = re.compile(r'ORD-\d{4,8}')

UNSUPPORTED_KEYS = {'pattern', 'minLength', 'maxLength', 'default', 'examples', '$schema'}

class TicketExtraction(BaseModel):
    customer_name: Optional[str] = Field(
        description='Full name of the customer if the text gives it. Otherwise null.'
    )
    order_id: Optional[str] = Field(
        description='Order id in the form ORD-48213. Add the ORD- prefix if it is missing. Null if the text has no order id.'
    )
    category: Category = Field(
        description='Main topic of the complaint. Use other when nothing fits.'
    )
    urgency: Urgency = Field(
        description='high if the customer has a deadline or is angry, medium if they want action soon, low otherwise.'
    )
    sentiment: Sentiment = Field(description='Overall mood of the customer.')
    amount_usd: Optional[float] = Field(
        ge=0,
        description='Money amount the customer mentions, as a plain number in US dollars. Null if there is none.',
    )
    summary: str = Field(
        min_length=5,
        max_length=200,
        description='One short sentence in English that says what went wrong.',
    )
    requested_action: Optional[str] = Field(
        description='What the customer wants us to do. Null if they do not say.'
    )

    @field_validator('order_id')
    @classmethod
    def check_order_id(cls, value):
        if value is None:
            return value
        if not ORDER_ID_PATTERN.fullmatch(value):
            raise ValueError('order_id must look like ORD-48213')
        return value

def strip_unsupported(node):
    if isinstance(node, dict):
        return {
            key: strip_unsupported(value)
            for key, value in node.items()
            if key not in UNSUPPORTED_KEYS
        }
    if isinstance(node, list):
        return [strip_unsupported(item) for item in node]
    return node

def to_api_schema(model):
    return strip_unsupported(model.model_json_schema())
