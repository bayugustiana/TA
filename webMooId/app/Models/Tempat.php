<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Tempat extends Model
{
    // Table Name
    protected $table = 'tempat';
    // Primary Key
    public $primary_key = 'id';
    // Timestamps
    public $timestamps = false;

    protected $fillable = [
        'id_peternakan',
        'jenis_tempat',
        'kapasitas_sapi'
    ];

    public function peternakan()
    {
        return $this->belongsTo('App\Models\Peternakan','id_peternakan');
    }

    public function sapi()
    {
        return $this->hasMany('App\Models\Sapi','id_tempat');
    }

    public function gateway()
    {
        return $this->hasOne('App\Models\Gateway','id_tempat');
    }
}
