# --- Code copy/pasted then adapted from
# https://github.com/wyhuai/DDNM/blob/main/functions/svd_operators.py
# The functions were modified to handle any SVD decomposed operators
# as long as it abides by the H_funcs API
import torch


def get_special_methods(A_funcs):
    """Get ``Lambda`` and ``Lambda_noise`` methods needed in DDNM."""
    Lambda_func = lambda *args: LmbdFuncs.Lambda(A_funcs, *args)
    Lambda_noise_func = lambda *args: LmbdFuncs.Lambda_noise(A_funcs, *args)

    return Lambda_func, Lambda_noise_func


class LmbdFuncs:

    def Lambda(self, vec, a, sigma_y, sigma_t, eta):

        Vt_vec = self.Vt(vec)
        _, dim = Vt_vec.shape

        singulars = self.singulars()
        lambda_t = torch.ones(dim, device=vec.device)
        temp_singulars = torch.zeros(dim, device=vec.device)
        temp_singulars[: singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1.0 / singulars
        inverse_singulars[singulars == 0] = 0.0

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
                singulars * sigma_t * (1 - eta**2) ** 0.5 / a / sigma_y
            )
        lambda_t = lambda_t.reshape(1, -1)
        Vt_vec = Vt_vec * lambda_t

        return self.V(Vt_vec)

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):

        Vt_vec = self.Vt(vec)
        Vt_eps = self.Vt(epsilon)
        _, dim = Vt_vec.shape

        singulars = self.singulars()
        d1_t = torch.ones(dim, device=vec.device) * sigma_t * eta
        d2_t = torch.ones(dim, device=vec.device) * sigma_t * (1 - eta**2) ** 0.5

        temp_singulars = torch.zeros(dim, device=vec.device)
        temp_singulars[: singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1.0 / singulars
        inverse_singulars[singulars == 0] = 0.0

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (sigma_t > a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(
                change_index * (sigma_t**2 - a**2 * sigma_y**2 * inverse_singulars**2)
            )
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = (
                d2_t * (-change_index + 1.0)
                + change_index * sigma_t * (1 - eta**2) ** 0.5
            )

        d1_t = d1_t.reshape(1, -1)
        d2_t = d2_t.reshape(1, -1)
        out_vec = Vt_vec * d1_t
        out_eps = Vt_eps * d2_t

        result_vec = self.V(out_vec)
        result_eps = self.V(out_eps)

        return result_vec + result_eps
