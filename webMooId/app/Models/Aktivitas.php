<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Aktivitas extends Model
{
    // Table Name
    protected $table = 'aktivitas';
    // Primary Key
    public $primary_key = 'id';
    // Timestamps
    public $timestamps = false;

    protected $fillable = [
        'id_sapi',
        'latitude',
        'longitude',
        'berat_badan',
        'suhu',
        'detak_jantung',
        'waktu'
    ];

    public function sapi()
    {
        return $this->belongsTo('App\Models\Sapi','id_sapi');
    }
}
