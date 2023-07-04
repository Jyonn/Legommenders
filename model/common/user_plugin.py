import torch
from UniTok import UniDep
from torch import nn

from loader.global_setting import Setting


class UserPlugin(nn.Module):
    attr_embeds: nn.ModuleDict
    empty_embed: nn.Parameter
    col_count: int
    project: nn.Sequential

    def __init__(self, depot: UniDep, hidden_size, select_cols=None):
        super().__init__()

        self.depot = depot
        self.hidden_size = hidden_size

        self.attr_embeds = nn.ModuleDict()
        self.empty_embed = nn.Parameter(torch.zeros(hidden_size))

        self.col_count = 0
        self.selected_cols = select_cols or self.depot.cols
        for col in self.selected_cols:
            if self.depot.id_col == col:
                continue

            voc = self.depot.cols[col].voc.name
            self.attr_embeds[voc] = nn.Embedding(
                num_embeddings=self.depot.cols[col].voc.size,
                embedding_dim=hidden_size
            )
            self.col_count += 1

        self.init_projection(hidden_size)
        self._device = None

        self.fast_eval = False
        self.fast_user_repr = None

    def init_projection(self, user_embed_size):
        self.project = nn.Sequential(
            nn.Linear(self.col_count * self.hidden_size + user_embed_size, user_embed_size),
            nn.ReLU(),
        )

    def start_fast_eval(self):
        self.fast_eval = True
        self.fast_user_repr = dict()

    def end_fast_eval(self):
        self.fast_eval = False
        self.fast_user_repr = None

    @property
    def device(self):
        if self._device:
            return self._device
        self._device = self.empty_embed.device
        return self._device

    def get_user_embedding(self, uid):
        if self.fast_eval and uid in self.fast_user_repr:
            return self.fast_user_repr[uid]

        attrs = self.depot[uid]
        values = []
        for attr in self.attr_embeds:
            value = attrs[attr]
            if not self.depot.cols[attr].list:
                value = [value]
            if value:
                value = torch.tensor(value).to(self.device)
                value = self.attr_embeds[attr](value).mean(dim=0)
            else:
                value = self.empty_embed
            values.append(value)
        # return torch.stack(values).mean(dim=0)
        user_embedding = torch.cat(values, dim=0)
        if self.fast_eval:
            self.fast_user_repr[uid] = user_embedding
        return user_embedding

    def forward(self, uids: torch.Tensor, user_embedding):
        user_embedding_dict = dict()
        user_embedding_list = []
        for uid in uids.cpu().tolist():
            if uid not in user_embedding_dict:
                user_embedding_dict[uid] = self.get_user_embedding(uid)
            user_embedding_list.append(user_embedding_dict[uid])
        plugged_embedding = torch.stack(user_embedding_list).to(self.device)
        return self.project(torch.cat([user_embedding, plugged_embedding], dim=1))
