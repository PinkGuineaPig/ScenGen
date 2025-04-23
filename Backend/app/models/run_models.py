# Backend/app/models/run_models.py
# ---------------------------
# Unified model definitions for model runs, configurations, and loss history
# ---------------------------

from datetime import datetime
from sqlalchemy.orm import backref, joinedload
from Backend.app import db

# ---------------------------
# ModelRunConfig: Stores all model-related hyperparameters set by the user
# ---------------------------
class ModelRunConfig(db.Model):
    __tablename__ = 'model_run_config'

    id          = db.Column(db.Integer, primary_key=True, nullable=False)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    model_type  = db.Column(db.String(50), nullable=False)
    parameters  = db.Column(db.JSON,    nullable=False)
    currency_pairs = db.Column(db.ARRAY(db.String(10)), nullable=False, default=list)

    # bring in your projection configs
    pca_configs = db.relationship(
        'PCAProjectionConfig',
        back_populates='config',
        lazy='select',
        cascade='all, delete-orphan'
    )
    som_configs = db.relationship(
        'SOMProjectionConfig',
        back_populates='config',
        lazy='select',
        cascade='all, delete-orphan'
    )
    runs = db.relationship(
        'ModelRun',
        back_populates='config',
        cascade='all, delete-orphan'
    )   



    # Backrefs provided by related models:
    #   .runs         -> list of ModelRun instances
    #   .loss_history -> list of ModelLossHistory instances

    def __repr__(self):
        return (
            f'<ModelRunConfig id={self.id} '
            f'type={self.model_type} pairs={self.currency_pairs}>'
        )

    def to_dict(self, include_relations=False):
        data = {
            'id':              self.id,
            'model_type':      self.model_type,
            'parameters':      self.parameters,
            'currency_pairs':  self.currency_pairs,
            'created_at':      self.created_at.isoformat(),
        }
        if include_relations:
            data['runs']        = [run.to_frontend_dict() for run in self.runs]
            data['pca_configs'] = [cfg.to_dict() for cfg in self.pca_configs]
            data['som_configs'] = [cfg.to_dict() for cfg in self.som_configs]
        return data

    @classmethod
    def get_all_model_configs(cls, session):
        """
        Fetch all configurations without loading relations.
        Use `config.runs` or `config.loss_history` to access linked data via backrefs.
        """
        return session.query(cls).all()
    
    # ----------------------------------------------------------------
    # 1) New instance method: emit one “flat” dict per sub-config pair
    # ----------------------------------------------------------------
    def to_flat_dicts(self):
        rows = []
        # if no pca/som, still emit at least one row
        pcas = self.pca_configs or [None]
        soms = self.som_configs or [None]

        # flatten your parameters JSON into columns
        # (if you know the exact keys, you can explicitly pull them out instead)
        flat_params = self.parameters.copy()

        for pca in pcas:
            for som in soms:
                row = {
                    'id':                self.id,
                    'model_type':        self.model_type,
                    **flat_params,       # each JSON key → its own column
                    'currency_pairs':    ','.join(self.currency_pairs),
                    'created_at':        self.created_at.isoformat(),
                    'pca_n_components':  getattr(pca, 'n_components', None),
                    'pca_whiten':        (pca.additional_params or {}).get('whiten') if pca else None,
                    'pca_solver':        (pca.additional_params or {}).get('solver') if pca else None,
                    'som_dims':          f"{som.x_dim}|{som.y_dim}" if som else None,
                    'som_iterations':    getattr(som, 'iterations', None),
                    'som_sigma':         (som.additional_params or {}).get('sigma') if som else None,
                    'som_learning_rate': (som.additional_params or {}).get('learning_rate') if som else None,
                }
                rows.append(row)

        return rows

    # ----------------------------------------------------------------------------
    # 2) New classmethod: eager-load everything and return one big flat list of rows
    # ----------------------------------------------------------------------------
    @classmethod
    def all_flat_dicts(cls, session):
        configs = (
            session.query(cls)
                   .options(
                      joinedload(cls.pca_configs),
                      joinedload(cls.som_configs),
                   )
                   .all()
        )
        flat = []
        for cfg in configs:
            flat.extend(cfg.to_flat_dicts())
        return flat

# ---------------------------
# ModelRun: Stores the trained model created by the user
# ---------------------------
class ModelRun(db.Model):
    __tablename__ = 'model_run'

    id          = db.Column(db.Integer, primary_key=True, nullable=False)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    config_id = db.Column(
        db.Integer,
        db.ForeignKey('model_run_config.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )

    config = db.relationship(
        'ModelRunConfig',
        back_populates='runs'
    )

    loss_history = db.relationship(
        'ModelLossHistory',
        back_populates='run',
        cascade='all, delete-orphan'
    )

    latent_points = db.relationship(
        'LatentPoint',
        back_populates='run',
        cascade='all, delete-orphan'
    )

    version     = db.Column(db.Integer, nullable=False)
    model_blob  = db.Column(db.LargeBinary, nullable=False)

    def __repr__(self):
        return f'<ModelRun id={self.id} v={self.version}>'

    def to_dict(self):
        """
        Full serialization including all fields (for internal use).
        """
        return {
            'id':         self.id,
            'config_id':  self.config_id,
            'version':    self.version,
            'created_at': self.created_at.isoformat(),
            'model_blob': self.model_blob,
        }

    def to_frontend_dict(self):
        """
        Lean serialization for client use—omits heavy/binary content.
        """
        return {
            'id':         self.id,
            'version':    self.version,
            'created_at': self.created_at.isoformat(),
        }

    @classmethod
    def get_all_models(cls, session, eager=False):
        """
        Fetch all model runs. Set `eager=True` to load configs in one step.
        """
        query = session.query(cls)
        if eager:
            query = query.options(joinedload(cls.config))
        return query.all()

# ---------------------------
# ModelLossHistory: Records loss/error for each epoch of a model run
# ---------------------------
class ModelLossHistory(db.Model):
    __tablename__ = 'model_loss_history'

    id           = db.Column(db.Integer, primary_key=True, nullable=False)
    model_run_id = db.Column(
        db.Integer,
        db.ForeignKey('model_run.id'),
        nullable=False,
        index=True
    )
    run = db.relationship(
        'ModelRun',
        back_populates='loss_history'
    )
     
    epoch        = db.Column(db.Integer, nullable=False)
    loss_type    = db.Column(db.String(50), nullable=False)
    value        = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f'<LossHistory run={self.model_run_id} epoch={self.epoch} type={self.loss_type}>'

    def to_dict(self):
        """
        Serialization for loss entries.
        """
        return {
            'id':            self.id,
            'model_run_id':  self.model_run_id,
            'epoch':         self.epoch,
            'loss_type':     self.loss_type,
            'value':         self.value,
        }
