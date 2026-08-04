from cobaya.likelihood import Likelihood
import likelihood


class DESI_ADPD_Likelihood(Likelihood):

    def get_requirements(self):
        return {}

    def get_can_support_params(self):
        return ["la_axis", "ba_axis", "om_k0"]

    def logp(self, **params_values):

        chi2 = likelihood.loglike(
            la_axis=params_values["la_axis"],
            ba_axis=params_values["ba_axis"],
            om_k0=params_values["om_k0"],
        )

        return -0.5 * chi2
