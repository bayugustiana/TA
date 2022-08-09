<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Artikel extends Model
{
    // Table Name
    protected $table = 'artikel';
    // Primary Key
    public $primary_key = 'id';
    // Timestamps
    public $timestamps = false;

    protected $fillable = [
        'id_dokter',
        'title',
        'slug',
        'konten',
        'waktu'
    ];

    public function dokter()
    {
        return $this->belongsTo('App\Models\Dokter','id_dokter');
    }
}
