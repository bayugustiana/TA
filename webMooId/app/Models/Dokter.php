<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Dokter extends Model
{
    // Table Name
    protected $table = 'dokter';
    // Primary Key
    public $primary_key = 'id';
    // Timestamps
    public $timestamps = false;

    protected $fillable = [
        'id_peternakan',
        'nama',
        'email',
        'no_telepon',
        'harga',
        'spesialis',
        'pendidikan_terakhir',
        'pengalaman_kerja',
        'avatar',
        'tempat_kerja'
    ];

    public function artikel()
    {
        return $this->hasMany('App\Models\Artikel','id_dokter');
    }

    public function rekam_medis()
    {
        return $this->hasMany('App\Models\Rekam_Medis','id_dokter');
    }
}
