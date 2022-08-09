<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Stok_Obat extends Model
{
    // Table Name
    protected $table = 'stok_obat';
    // Primary Key
    public $primary_key = 'id';
    // Timestamps
    public $timestamps = false;

    protected $fillable = [
        'id_obat',
        'tanggal_kadaluarsa'
    ];

    public function obat()
    {
        return $this->belongsTo('App\Models\Obat','id_obat');
    }
}
