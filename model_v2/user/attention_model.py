from model_v2.common.attention_fusion_model import AttentionFusionOperator
from model_v2.interaction.negative_sampling import NegativeSampling
from model_v2.user.base_model import BaseUserModel


class UserAttentionFusionModel(BaseUserModel):
    operator_class = AttentionFusionOperator
    interaction = NegativeSampling

    def forward(self, clicks, **kwargs):
        candidates = kwargs['candidates']
        user_embedding = self.operator(clicks)
        return self.interaction.predict(user_embedding, candidates, labels=None)
