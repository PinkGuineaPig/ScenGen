from Backend.app import db
from datetime import datetime

# ---------------------------
# LatentPoint: Stores latent-space vectors for each input sequence snapshot.
# ---------------------------
class LatentPoint(db.Model):
    __tablename__ = 'latent_point'

    id            = db.Column(db.Integer, primary_key=True)
    model_run_id  = db.Column(db.Integer, db.ForeignKey('model_run.id'), nullable=False, index=True)
    start_date    = db.Column(db.Date, nullable=False)
    lag           = db.Column(db.Integer, nullable=False)
    latent_vector = db.Column(db.ARRAY(db.Float), nullable=True)

    __table_args__ = (
        db.UniqueConstraint('model_run_id', 'start_date', 'lag', name='uq_latent_seq'),
    )

    # Relationship back to the run for convenience
    run = db.relationship(
        'ModelRun',
        backref=db.backref('latent_points', lazy=True, cascade='all, delete-orphan')
    )

    def get_pca_vector(self, session, config_id=None):
        from .latent_models import PCAProjection
        query = session.query(PCAProjection).filter_by(latent_point_id=self.id)
        if config_id is not None:
            query = query.filter_by(config_id=config_id)
        return [p.value for p in query.order_by(PCAProjection.dim).all()]

    def get_som_coordinates(self, session, config_id=None):
        from .latent_models import SOMProjection
        query = session.query(SOMProjection).filter_by(latent_point_id=self.id)
        if config_id is not None:
            query = query.filter_by(config_id=config_id)
        return query.first()


# ---------------------------
# LatentInput: Links each latent point back to its original data source.
# ---------------------------
class LatentInput(db.Model):
    __tablename__ = 'latent_input'

    id              = db.Column(db.Integer, primary_key=True)
    latent_point_id = db.Column(db.Integer, db.ForeignKey('latent_point.id'), nullable=False, index=True)
    currency_pair   = db.Column(db.String(10), nullable=False)
    date            = db.Column(db.Date, nullable=False)
    source_table    = db.Column(db.String(80), nullable=False)

    latent_point = db.relationship(
        'LatentPoint',
        backref=db.backref('inputs', lazy=True, cascade='all, delete-orphan')
    )

    def __repr__(self):
        return f"<LatentInput point={self.latent_point_id} pair={self.currency_pair} date={self.date}>"


# ---------------------------
# SOMProjectionConfig: Stores configuration details for SOM projections.
# ---------------------------
class SOMProjectionConfig(db.Model):
    __tablename__ = 'som_projection_config'

    id            = db.Column(db.Integer, primary_key=True)
    model_run_id  = db.Column(db.Integer, db.ForeignKey('model_run.id'), nullable=False, index=True)
    x_dim         = db.Column(db.Integer, nullable=False)
    y_dim         = db.Column(db.Integer, nullable=False)
    iterations    = db.Column(db.Integer, nullable=False)
    additional_params = db.Column(db.JSON, nullable=True)

    model_run = db.relationship(
        'ModelRun',
        backref=db.backref('som_configs', lazy=True, cascade='all, delete-orphan')
    )

    def __repr__(self):
        return (
            f'<SOMConfig run={self.model_run_id} '
            f'{self.x_dim}x{self.y_dim} iters={self.iterations}>'
        )


# ---------------------------
# SOMProjection: Maps latent points onto a 2D SOM grid cell.
# ---------------------------
class SOMProjection(db.Model):
    __tablename__ = 'som_projection'

    id              = db.Column(db.Integer, primary_key=True)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    latent_point_id = db.Column(db.Integer, db.ForeignKey('latent_point.id'), nullable=False, index=True)
    config_id       = db.Column(db.Integer, db.ForeignKey('som_projection_config.id'), nullable=False, index=True)
    x               = db.Column(db.Integer, nullable=False)
    y               = db.Column(db.Integer, nullable=False)

    @classmethod
    def get_points_for_cell(cls, session, x, y, config_id=None):
        query = session.query(cls).filter_by(x=x, y=y)
        if config_id is not None:
            query = query.filter_by(config_id=config_id)
        return query.all()


# ---------------------------
# PCAProjectionConfig: Stores configuration details for PCA projections.
# ---------------------------
class PCAProjectionConfig(db.Model):
    __tablename__ = 'pca_projection_config'

    id            = db.Column(db.Integer, primary_key=True)
    model_run_id  = db.Column(db.Integer, db.ForeignKey('model_run.id'), nullable=False, index=True)
    n_components  = db.Column(db.Integer, nullable=False)
    additional_params = db.Column(db.JSON, nullable=True)

    model_run = db.relationship(
        'ModelRun',
        backref=db.backref('pca_configs', lazy=True, cascade='all, delete-orphan')
    )

    def __repr__(self):
        return f'<PCAConfig run={self.model_run_id} components={self.n_components}>'


# ---------------------------
# PCAProjection: Stores individual PCA coordinate values for latent points.
# ---------------------------
class PCAProjection(db.Model):
    __tablename__ = 'pca_projection'

    id              = db.Column(db.Integer, primary_key=True)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    latent_point_id = db.Column(db.Integer, db.ForeignKey('latent_point.id'), nullable=False, index=True)
    config_id       = db.Column(db.Integer, db.ForeignKey('pca_projection_config.id'), nullable=False, index=True)
    dim             = db.Column(db.Integer, nullable=False)
    value           = db.Column(db.Float, nullable=False)

    @classmethod
    def get_vectors_for_config(cls, session, config_id):
        query = session.query(cls).filter_by(config_id=config_id)
        results = query.order_by(cls.latent_point_id, cls.dim).all()
        vectors = {}
        for p in results:
            vectors.setdefault(p.latent_point_id, []).append(p.value)
        return vectors
