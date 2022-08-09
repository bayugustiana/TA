<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Petugas extends Model
{
    // Table Name
    protected $table = 'petugas';
    // Primary Key
    public $primary_key = 'no_ktp';
    // Timestamps
    public $timestamps = false;

    protected $fillable = [
        'id_peternakan',
        'nama_petugas',
        'username',
        'password'
    ];

    public function peternakan()
    {
        return $this->belongsTo('App\Models\Peternakan','id_peternakan');
    }

    public function task()
    {
        return $this->hasMany('App\Models\Task','no_ktp');
    }
}
