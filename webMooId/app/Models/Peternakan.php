<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Peternakan extends Model
{
    // Table Name
    protected $table = 'peternakan';
    // Primary Key
    public $primary_key = 'id';
    // Timestamps
    public $timestamps = false;

    protected $fillable = [
        'nama_peternakan',
        'username',
        'password',
        'jenis_peternakan',
        'nama_lengkap',
        'no_telepon',
        'email',
        'alamat_lengkap',
        'latitude',
        'longitude',
        'area',
        'avatar'
    ];

    public function sapi()
    {
        return $this->hasManyThrough(
            'App\Models\Sapi',
            'App\Models\Tempat',
            'id_peternakan',
            'id_tempat',
            'id',
            'id'
        );
    }
    public function artikel()
    {
        return $this->hasManyThrough(
            'App\Models\Artikel',
            'App\Models\Dokter',
            'id_peternakan',
            'id_dokter',
            'id',
            'id'
        );
    }

    public function tempat()
    {
        return $this->hasMany('App\Models\Tempat','id_peternakan');
    }

    public function transaksi()
    {
        return $this->hasMany('App\Models\Transaksi','id_peternakan');
    }

    public function dokter()
    {
        return $this->hasMany('App\Models\Dokter','id_peternakan');
    }

    public function petugas()
    {
        return $this->hasMany('App\Models\Petugas','id_peternakan');
    }
}
