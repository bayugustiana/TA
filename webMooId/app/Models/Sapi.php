<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Sapi extends Model
{
    // Table Name
    protected $table = 'sapi';
    // Primary Key
    public $primary_key = 'id';
    // Timestamps
    public $timestamps = false;

    protected $fillable = [
        'id_tempat',
        'tanggal_lahir',
        'jenis_sapi',
        'tipe_sapi',
        'jenis_kelamin'
    ];

    public function aktivitas()
    {
        return $this->hasMany('App\Models\Aktivitas','id_sapi');
    }

    public function tempat()
    {
        return $this->belongsTo('App\Models\Tempat','id_tempat');
    }
}
