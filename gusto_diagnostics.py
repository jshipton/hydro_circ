from gusto.diagnostics import DiagnosticField
from gusto_physics import w, evap, precip


class WDiagnostic(DiagnosticField):

    name = "vertical_velocity"

    def __init__(self, parameters, required_fields=('precipitation',)):

        self.parameters = parameters

        super().__init__(required_fields=required_fields)

    def setup(self, domain, state_fields):

        self.expr = w(self.parameters, state_fields('precipitation'))

        super().setup(domain, state_fields)


class EvaporationDiagnostic(DiagnosticField):

    name = "evaporation"

    def __init__(self, parameters, required_fields=('water_vapour', 'u')):

        self.parameters = parameters

        super().__init__(required_fields=required_fields)

    def setup(self, domain, state_fields):

        self.expr = evap(self.parameters, state_fields('water_vapour'),
                         state_fields('u'), self.qs)

        super().setup(domain, state_fields)


class PrecipitationDiagnostic(DiagnosticField):

    name = "precipitation"

    def __init__(self, parameters, required_fields=('water_vapour',)):

        self.parameters = parameters

        super().__init__(required_fields=required_fields)

    def setup(self, domain, state_fields):

        self.expr = precip(self.parameters, state_fields('water_vapour'))

        super().setup(domain, state_fields)
