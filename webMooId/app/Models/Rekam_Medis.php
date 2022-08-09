<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Rekam_Medis extends Model
{
    // Table Name
    protected $table = 'rekam_medis';
    // Primary Key
    public $primary_key = 'id';
    // Timestamps
    public $timestamps = false;

    protected $fillable = [
        'id_obat',
        'id_dokter',
        'id_sapi',
        'nama_pesan',
        'waktu'
    ];

    public function obat()
    {
        return $this->belongsTo('App\Models\Obat','id_obat');
    }

    public function dokter()
    {
        return $this->belongsTo('App\Models\Dokter','id_dokter');
    }

    public function sapi()
    {
        return $this->belongsTo('App\Models\Sapi','id_sapi');
    }
}
