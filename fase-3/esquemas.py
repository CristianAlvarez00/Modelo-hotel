"""Las 17 claves son obligatorias; null representa un dato faltante."""

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, StrictStr

Numero = StrictInt | StrictFloat | None
Texto = StrictStr | None


class Reserva(BaseModel):
    # La conversión de CSV permite texto numérico; por HTTP exigimos números JSON.
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    lead_time: Numero
    stays_in_weekend_nights: Numero
    stays_in_week_nights: Numero
    adults: Numero
    children: Numero
    babies: Numero
    is_repeated_guest: Numero
    previous_cancellations: Numero
    previous_bookings_not_canceled: Numero
    hotel: Texto
    arrival_date_month: Texto
    meal: Texto
    country: Texto
    market_segment: Texto
    distribution_channel: Texto
    reserved_room_type: Texto
    customer_type: Texto


class SolicitudEntrenamiento(BaseModel):
    model_config = ConfigDict(extra="forbid")
