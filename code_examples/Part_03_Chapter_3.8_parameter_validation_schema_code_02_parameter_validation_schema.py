from pydantic import BaseModel, Field, EmailStr, validator
from typing import Literal

class TransferFundsParams(BaseModel):
    """Parameter schema for transfer_funds tool with comprehensive validation"""

    from_account: str = Field(..., min_length=8, max_length=12,
                               description="Source account number")
    to_account: str = Field(..., min_length=8, max_length=12,
                            description="Destination account number")
    amount: float = Field(..., gt=0, le=10000,
                          description="Transfer amount, must be positive and ≤$10,000")
    currency: Literal["USD", "EUR", "GBP"] = Field(default="USD",
                                                    description="Currency code")
    reference: str = Field(None, max_length=140,
                           description="Optional transfer reference")

    @validator('to_account')
    def accounts_must_differ(cls, v, values):
        """Prevent transfers to same account"""
        if 'from_account' in values and v == values['from_account']:
            raise ValueError('Cannot transfer to same account')
        return v

    @validator('amount')
    def amount_precision(cls, v):
        """Ensure monetary precision (max 2 decimal places)"""
        if round(v, 2) != v:
            raise ValueError('Amount must have at most 2 decimal places')
        return v

def validate_and_invoke_transfer(agent_output: dict) -> dict:
    """Validate parameters before executing transfer"""
    try:
        # Parse and validate agent output against schema
        params = TransferFundsParams(**agent_output)

        # If validation succeeds, parameters are guaranteed correct format
        result = execute_transfer(
            from_account=params.from_account,
            to_account=params.to_account,
            amount=params.amount,
            currency=params.currency,
            reference=params.reference
        )
        return {"status": "success", "result": result}

    except ValidationError as e:
        # Log parameter validation failure for action accuracy tracking
        log_parameter_error(agent_output, str(e))
        return {"status": "validation_failed", "errors": e.errors()}